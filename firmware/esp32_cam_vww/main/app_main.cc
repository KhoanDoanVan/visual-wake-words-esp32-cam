#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "driver/gpio.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "img_converters.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "app_config.h"
#include "board_camera_pins.h"

namespace {
constexpr char kTag[] = "vww";
constexpr int kInputWidth = 96;
constexpr int kInputHeight = 96;
constexpr int kInputChannels = 3;
constexpr int kCaptureWidth = 160;
constexpr int kCaptureHeight = 120;
constexpr size_t kTensorArenaBytes = 350 * 1024;
constexpr size_t kRgbScratchBytes = kCaptureWidth * kCaptureHeight * kInputChannels;

class WakeDebouncer {
 public:
  bool Update(bool detected) {
    history_[cursor_] = detected;
    cursor_ = (cursor_ + 1) % history_.size();
    if (samples_ < history_.size()) ++samples_;
    int positives = 0;
    for (size_t i = 0; i < samples_; ++i) positives += history_[i] ? 1 : 0;
    if (!active_ && samples_ == history_.size() && positives >= kWakeFramesRequired) {
      active_ = true;
    } else if (active_ && positives <= kReleaseFramesMaximum) {
      active_ = false;
    }
    return active_;
  }

 private:
  std::array<bool, kDebounceWindowFrames> history_{};
  size_t cursor_ = 0;
  size_t samples_ = 0;
  bool active_ = false;
};

int8_t Quantize(uint8_t value, float scale, int zero_point) {
  const int quantized = static_cast<int>(std::lround(value / scale)) + zero_point;
  return static_cast<int8_t>(std::clamp(quantized, -128, 127));
}

bool ResizeRgb565ToInt8(const camera_fb_t* frame, uint8_t* rgb888, TfLiteTensor* input) {
  if (frame->format != PIXFORMAT_RGB565 || frame->width != kCaptureWidth ||
      frame->height != kCaptureHeight) {
    ESP_LOGE(kTag, "Unexpected frame: %ux%u format=%d", static_cast<unsigned>(frame->width),
             static_cast<unsigned>(frame->height), static_cast<int>(frame->format));
    return false;
  }
  if (!fmt2rgb888(frame->buf, frame->len, frame->format, rgb888)) return false;
  const float scale = input->params.scale;
  const int zero = input->params.zero_point;
  for (int y = 0; y < kInputHeight; ++y) {
    const float source_y = std::clamp(
        (y + 0.5f) * frame->height / kInputHeight - 0.5f, 0.0f,
        static_cast<float>(frame->height - 1));
    const int y0 = static_cast<int>(source_y);
    const int y1 = std::min(y0 + 1, static_cast<int>(frame->height - 1));
    const float wy = source_y - y0;
    for (int x = 0; x < kInputWidth; ++x) {
      const float source_x = std::clamp(
          (x + 0.5f) * frame->width / kInputWidth - 0.5f, 0.0f,
          static_cast<float>(frame->width - 1));
      const int x0 = static_cast<int>(source_x);
      const int x1 = std::min(x0 + 1, static_cast<int>(frame->width - 1));
      const float wx = source_x - x0;
      const int output_offset = (y * kInputWidth + x) * kInputChannels;
      for (int channel = 0; channel < kInputChannels; ++channel) {
        const float top =
            rgb888[(y0 * frame->width + x0) * kInputChannels + channel] * (1.0f - wx) +
            rgb888[(y0 * frame->width + x1) * kInputChannels + channel] * wx;
        const float bottom =
            rgb888[(y1 * frame->width + x0) * kInputChannels + channel] * (1.0f - wx) +
            rgb888[(y1 * frame->width + x1) * kInputChannels + channel] * wx;
        input->data.int8[output_offset + channel] = Quantize(
            static_cast<uint8_t>(std::lround(top * (1.0f - wy) + bottom * wy)), scale, zero);
      }
    }
  }
  return true;
}

bool ValidateTensorContract(const TfLiteTensor* input, const TfLiteTensor* output) {
  const bool input_ok = input != nullptr && input->type == kTfLiteInt8 && input->dims->size == 4 &&
                        input->dims->data[0] == 1 && input->dims->data[1] == kInputHeight &&
                        input->dims->data[2] == kInputWidth &&
                        input->dims->data[3] == kInputChannels;
  const bool output_ok = output != nullptr && output->type == kTfLiteInt8 &&
                         output->bytes == sizeof(int8_t);
  if (!input_ok || !output_ok) {
    ESP_LOGE(kTag, "Model tensor contract mismatch");
    return false;
  }
  ESP_LOGI(kTag, "Input INT8 scale=%.6f zero=%d bytes=%u", input->params.scale,
           static_cast<int>(input->params.zero_point), static_cast<unsigned>(input->bytes));
  ESP_LOGI(kTag, "Output INT8 scale=%.6f zero=%d bytes=%u", output->params.scale,
           static_cast<int>(output->params.zero_point), static_cast<unsigned>(output->bytes));
  return true;
}

void LogMemory(const char* phase) {
  ESP_LOGI(kTag, "%s: internal_free=%u internal_largest=%u psram_free=%u psram_largest=%u", phase,
           static_cast<unsigned>(
               heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(
               heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(
               heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
}

bool InitializeWakeOutput() {
  gpio_config_t output{};
  output.pin_bit_mask = 1ULL << kWakeOutputGpio;
  output.mode = GPIO_MODE_OUTPUT;
  output.pull_up_en = GPIO_PULLUP_DISABLE;
  output.pull_down_en = GPIO_PULLDOWN_DISABLE;
  output.intr_type = GPIO_INTR_DISABLE;
  if (gpio_config(&output) != ESP_OK) return false;
  gpio_set_level(kWakeOutputGpio, kWakeOutputActiveLow ? 1 : 0);
  return true;
}

void SetWakeOutput(bool active) {
  gpio_set_level(kWakeOutputGpio, kWakeOutputActiveLow ? !active : active);
}
}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "VWW startup: model=%u bytes threshold=%.2f debounce=%d/%d",
           g_vww_model_data_len, kPersonThreshold, kWakeFramesRequired, kDebounceWindowFrames);
  if (!esp_psram_is_initialized()) {
    ESP_LOGE(kTag, "PSRAM is not initialized; this build requires an ESP32-CAM with PSRAM");
    return;
  }
  if (!InitializeWakeOutput()) {
    ESP_LOGE(kTag, "Wake output GPIO initialization failed");
    return;
  }
  LogMemory("before allocations");

  camera_config_t camera = MakeBoardCameraConfig(PIXFORMAT_RGB565, FRAMESIZE_QQVGA);
  const esp_err_t camera_result = esp_camera_init(&camera);
  if (camera_result != ESP_OK) {
    ESP_LOGE(kTag, "Camera init failed: 0x%x", camera_result);
    return;
  }

  const tflite::Model* model = tflite::GetModel(g_vww_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "TFLite schema mismatch: model=%lu runtime=%d",
             static_cast<unsigned long>(model->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddReshape();

  auto* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(
      16, kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  auto* rgb888 = static_cast<uint8_t*>(
      heap_caps_malloc(kRgbScratchBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (arena == nullptr || rgb888 == nullptr) {
    ESP_LOGE(kTag, "PSRAM allocation failed: arena=%p rgb=%p", arena, rgb888);
    return;
  }

  tflite::MicroInterpreter interpreter(model, resolver, arena, kTensorArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed; increase kTensorArenaBytes");
    return;
  }
  TfLiteTensor* input = interpreter.input(0);
  const TfLiteTensor* output = interpreter.output(0);
  if (!ValidateTensorContract(input, output)) return;
  LogMemory("ready");

  WakeDebouncer debounce;
  bool previous_wake = false;
  uint32_t frame_number = 0;
  while (true) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ESP_LOGW(kTag, "Camera capture failed");
      vTaskDelay(pdMS_TO_TICKS(kFrameIntervalMs));
      continue;
    }
    const bool converted = ResizeRgb565ToInt8(frame, rgb888, input);
    esp_camera_fb_return(frame);
    if (!converted) {
      ESP_LOGE(kTag, "RGB565 conversion failed");
      continue;
    }

    const int64_t started_us = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
      ESP_LOGE(kTag, "Inference failed");
      continue;
    }
    const int64_t latency_us = esp_timer_get_time() - started_us;
    output = interpreter.output(0);
    const float probability =
        (static_cast<int>(output->data.int8[0]) - output->params.zero_point) * output->params.scale;
    const bool wake = debounce.Update(probability >= kPersonThreshold);
    SetWakeOutput(wake);
    if (wake != previous_wake) {
      ESP_LOGW(kTag, "WAKE=%s probability=%.3f", wake ? "ON" : "OFF", probability);
      previous_wake = wake;
    }
    if ((frame_number++ % kLogEveryFrames) == 0) {
      ESP_LOGI(kTag, "p=%.3f wake=%d inference=%.1fms", probability, wake,
               latency_us / 1000.0f);
    }
    vTaskDelay(pdMS_TO_TICKS(kFrameIntervalMs));
  }
}
