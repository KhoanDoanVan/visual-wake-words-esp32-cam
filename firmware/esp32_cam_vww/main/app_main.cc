#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

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
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "app_config.h"
#include "board_camera_pins.h"
#include "web_server.h"

namespace {
constexpr char kTag[] = "vww";
#ifndef VWW_INPUT_SIZE
#define VWW_INPUT_SIZE 80
#endif
#ifndef VWW_MODEL_VARIANT
#define VWW_MODEL_VARIANT "fast_80"
#endif
constexpr int kInputWidth = VWW_INPUT_SIZE;
constexpr int kInputHeight = VWW_INPUT_SIZE;
constexpr int kInputChannels = 3;
constexpr int kCaptureWidth = 160;
constexpr int kCaptureHeight = 120;
constexpr size_t kTensorArenaBytes = 350 * 1024;
constexpr size_t kRgbScratchBytes = kCaptureWidth * kCaptureHeight * kInputChannels;
constexpr size_t kInputBytes = kInputWidth * kInputHeight * kInputChannels;
constexpr float kMinimumColorbarMean = 60.0f;
constexpr float kMinimumLiveMean = 30.0f;
// Despite its name, esp32-camera's fmt2rgb888() writes B, G, R.
// The TensorFlow model was trained with tf.io.decode_jpeg(..., channels=3),
// which produces R, G, B.
constexpr std::array<int, kInputChannels> kBgrToRgbChannel = {2, 1, 0};

// Real-camera domain correction selected against the supplied OV3660 stream.
// First reduce chroma by 50% around BT.601 luma, then apply gamma 1.2. The
// fixed-point luma and 256-byte LUT avoid per-pixel powf(), require no extra
// frame buffer, and keep the model itself unchanged.
constexpr std::array<uint8_t, 256> kGamma12Lut = {
    0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9,
    9, 10, 11, 11, 12, 13, 13, 14, 15, 16, 16, 17, 18, 19, 20, 20,
    21, 22, 23, 24, 24, 25, 26, 27, 28, 28, 29, 30, 31, 32, 33, 34,
    34, 35, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145,
    146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 159, 160, 161, 162,
    163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180,
    181, 183, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 199,
    200, 201, 202, 203, 204, 205, 207, 208, 209, 210, 211, 212, 214, 215, 216, 217,
    218, 219, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 234, 235, 236,
    237, 238, 239, 241, 242, 243, 244, 245, 247, 248, 249, 250, 251, 253, 254, 255,
};

struct InputStats {
  std::array<float, kInputChannels> mean{};
  uint8_t minimum = 255;
  uint8_t maximum = 0;
  float mean_absolute_delta = -1.0f;
  uint32_t checksum = 2166136261u;
};

// TFLite Micro invokes the profiler once per fused FlatBuffer operator. This
// fixed-size implementation records aggregate microsecond timings without the
// roughly 80 KiB event buffer used by the generic MicroProfiler.
class DeviceOperatorProfiler final : public tflite::MicroProfilerInterface {
 public:
  void BeginInference() {
    current_operator_ = 0;
    collecting_ = completed_invocations_ >= kProfileWarmupInvocations &&
                  measured_invocations_ < kProfileMeasuredInvocations;
    if (collecting_) inference_started_us_ = esp_timer_get_time();
  }

  uint32_t BeginEvent(const char* tag) override {
    const uint32_t index = current_operator_++;
    if (index >= kMaximumOperators || !collecting_) return index;
    if (tags_[index] == nullptr) tags_[index] = tag;
    operator_started_us_[index] = esp_timer_get_time();
    return index;
  }

  void EndEvent(uint32_t event_handle) override {
    if (!collecting_ || event_handle >= kMaximumOperators) return;
    const uint32_t elapsed_us =
        static_cast<uint32_t>(esp_timer_get_time() - operator_started_us_[event_handle]);
    OperatorTiming& timing = timings_[event_handle];
    timing.total_us += elapsed_us;
    timing.minimum_us = std::min(timing.minimum_us, elapsed_us);
    timing.maximum_us = std::max(timing.maximum_us, elapsed_us);
    ++timing.count;
  }

  void EndInference() {
    if (collecting_) {
      const uint32_t elapsed_us =
          static_cast<uint32_t>(esp_timer_get_time() - inference_started_us_);
      total_inference_us_ += elapsed_us;
      minimum_inference_us_ = std::min(minimum_inference_us_, elapsed_us);
      maximum_inference_us_ = std::max(maximum_inference_us_, elapsed_us);
      operator_count_ = std::max(operator_count_, current_operator_);
      ++measured_invocations_;
    }
    ++completed_invocations_;
  }

  bool ReadyToReport() const {
    return measured_invocations_ == kProfileMeasuredInvocations && !reported_;
  }

  void LogReport(size_t arena_used_bytes) {
    if (!ReadyToReport()) return;
    reported_ = true;
    ESP_LOGI(kTag,
             "VWW_PROFILE,BEGIN,variant=%s,input=%dx%dx%d,batch=1,samples=%u,"
             "warmup=%d,arena_used_bytes=%u",
             VWW_MODEL_VARIANT, kInputWidth, kInputHeight, kInputChannels,
             static_cast<unsigned>(measured_invocations_), kProfileWarmupInvocations,
             static_cast<unsigned>(arena_used_bytes));
    uint64_t operator_total_us = 0;
    for (uint32_t index = 0; index < std::min(operator_count_, kMaximumOperators); ++index) {
      const OperatorTiming& timing = timings_[index];
      operator_total_us += timing.total_us;
      ESP_LOGI(kTag,
               "VWW_PROFILE,OP,index=%u,tag=%s,count=%u,total_us=%llu,mean_us=%.2f,"
               "min_us=%u,max_us=%u",
               static_cast<unsigned>(index), tags_[index] == nullptr ? "UNKNOWN" : tags_[index],
               static_cast<unsigned>(timing.count),
               static_cast<unsigned long long>(timing.total_us),
               timing.count == 0 ? 0.0 : static_cast<double>(timing.total_us) / timing.count,
               static_cast<unsigned>(timing.minimum_us == UINT32_MAX ? 0 : timing.minimum_us),
               static_cast<unsigned>(timing.maximum_us));
    }
    ESP_LOGI(kTag,
             "VWW_PROFILE,TOTAL,count=%u,total_us=%llu,mean_us=%.2f,min_us=%u,max_us=%u,"
             "operator_sum_mean_us=%.2f,unprofiled_mean_us=%.2f",
             static_cast<unsigned>(measured_invocations_),
             static_cast<unsigned long long>(total_inference_us_),
             static_cast<double>(total_inference_us_) / measured_invocations_,
             static_cast<unsigned>(minimum_inference_us_),
             static_cast<unsigned>(maximum_inference_us_),
             static_cast<double>(operator_total_us) / measured_invocations_,
             static_cast<double>(total_inference_us_ - operator_total_us) /
                 measured_invocations_);
    ESP_LOGI(kTag, "VWW_PROFILE,END");
  }

 private:
  static constexpr uint32_t kMaximumOperators = 64;
  struct OperatorTiming {
    uint64_t total_us = 0;
    uint32_t minimum_us = UINT32_MAX;
    uint32_t maximum_us = 0;
    uint32_t count = 0;
  };

  std::array<const char*, kMaximumOperators> tags_{};
  std::array<int64_t, kMaximumOperators> operator_started_us_{};
  std::array<OperatorTiming, kMaximumOperators> timings_{};
  uint32_t current_operator_ = 0;
  uint32_t operator_count_ = 0;
  uint32_t completed_invocations_ = 0;
  uint32_t measured_invocations_ = 0;
  int64_t inference_started_us_ = 0;
  uint64_t total_inference_us_ = 0;
  uint32_t minimum_inference_us_ = UINT32_MAX;
  uint32_t maximum_inference_us_ = 0;
  bool collecting_ = false;
  bool reported_ = false;
};

class WakeDebouncer {
 public:
  bool Update(bool detected, bool activation_allowed) {
    // Motion gates only the inactive -> active transition. Once active, retain
    // normal classifier votes so a stationary person remains detected.
    history_[cursor_] = active_ ? detected : (detected && activation_allowed);
    cursor_ = (cursor_ + 1) % history_.size();
    if (samples_ < history_.size()) ++samples_;
    positives_ = 0;
    for (size_t i = 0; i < samples_; ++i) positives_ += history_[i] ? 1 : 0;
    if (!active_ && positives_ >= kWakeFramesRequired) {
      active_ = true;
    } else if (active_ && positives_ <= kReleaseFramesMaximum) {
      Reset();
    }
    return active_;
  }

  int PositiveVotes() const { return positives_; }
  int SampleCount() const { return static_cast<int>(samples_); }

  void Reset() {
    history_.fill(false);
    cursor_ = 0;
    samples_ = 0;
    positives_ = 0;
    active_ = false;
  }

 private:
  std::array<bool, kDebounceWindowFrames> history_{};
  size_t cursor_ = 0;
  size_t samples_ = 0;
  int positives_ = 0;
  bool active_ = false;
};

int8_t Quantize(uint8_t value, float scale, int zero_point) {
  const int quantized = static_cast<int>(std::lround(value / scale)) + zero_point;
  return static_cast<int8_t>(std::clamp(quantized, -128, 127));
}

InputStats SummarizeInput(const TfLiteTensor* input, const int8_t* previous,
                          bool has_previous) {
  InputStats stats;
  std::array<uint64_t, kInputChannels> channel_sums{};
  uint64_t absolute_delta_sum = 0;
  for (size_t index = 0; index < kInputBytes; ++index) {
    const int quantized = static_cast<int>(input->data.int8[index]);
    const int value = std::clamp(
        static_cast<int>(std::lround((quantized - input->params.zero_point) * input->params.scale)),
        0, 255);
    channel_sums[index % kInputChannels] += value;
    stats.minimum = std::min(stats.minimum, static_cast<uint8_t>(value));
    stats.maximum = std::max(stats.maximum, static_cast<uint8_t>(value));
    stats.checksum ^= static_cast<uint8_t>(input->data.int8[index]);
    stats.checksum *= 16777619u;
    if (has_previous) {
      absolute_delta_sum += std::abs(quantized - static_cast<int>(previous[index]));
    }
  }
  constexpr float kPixels = kInputWidth * kInputHeight;
  for (int channel = 0; channel < kInputChannels; ++channel) {
    stats.mean[channel] = channel_sums[channel] / kPixels;
  }
  if (has_previous) {
    stats.mean_absolute_delta =
        absolute_delta_sum * input->params.scale / static_cast<float>(kInputBytes);
  }
  return stats;
}

void LogAsciiPreview(const TfLiteTensor* input) {
  constexpr char kRamp[] = " .:-=+*#%@";
  constexpr int kPreviewWidth = 32;
  constexpr int kPreviewHeight = 24;
  char line[kPreviewWidth + 1]{};
  ESP_LOGI(kTag, "Input preview (dark -> bright):");
  for (int preview_y = 0; preview_y < kPreviewHeight; ++preview_y) {
    const int source_y = preview_y * kInputHeight / kPreviewHeight;
    for (int preview_x = 0; preview_x < kPreviewWidth; ++preview_x) {
      const int source_x = preview_x * kInputWidth / kPreviewWidth;
      const int offset = (source_y * kInputWidth + source_x) * kInputChannels;
      int luminance = 0;
      for (int channel = 0; channel < kInputChannels; ++channel) {
        const int quantized = static_cast<int>(input->data.int8[offset + channel]);
        luminance += std::clamp(
            static_cast<int>(std::lround(
                (quantized - input->params.zero_point) * input->params.scale)),
            0, 255);
      }
      luminance /= kInputChannels;
      line[preview_x] = kRamp[luminance * (sizeof(kRamp) - 2) / 255];
    }
    std::puts(line);
  }
}

bool ResizeFrameToInt8(const camera_fb_t* frame, uint8_t* bgr888, TfLiteTensor* input) {
  if (frame->format != PIXFORMAT_JPEG || frame->width != kCaptureWidth ||
      frame->height != kCaptureHeight) {
    ESP_LOGE(kTag, "Unexpected frame: %ux%u format=%d", static_cast<unsigned>(frame->width),
             static_cast<unsigned>(frame->height), static_cast<int>(frame->format));
    return false;
  }
  if (!fmt2rgb888(frame->buf, frame->len, frame->format, bgr888)) return false;
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
      std::array<uint8_t, kInputChannels> rgb{};
      for (int channel = 0; channel < kInputChannels; ++channel) {
        const int source_channel = kBgrToRgbChannel[channel];
        const float top =
            bgr888[(y0 * frame->width + x0) * kInputChannels + source_channel] * (1.0f - wx) +
            bgr888[(y0 * frame->width + x1) * kInputChannels + source_channel] * wx;
        const float bottom =
            bgr888[(y1 * frame->width + x0) * kInputChannels + source_channel] * (1.0f - wx) +
            bgr888[(y1 * frame->width + x1) * kInputChannels + source_channel] * wx;
        rgb[channel] =
            static_cast<uint8_t>(std::lround(top * (1.0f - wy) + bottom * wy));
      }

      // Integer BT.601 weights sum to 256. Averaging each RGB channel with
      // luma implements saturation=0.5 with one add and shift per channel.
      const int luma =
          (77 * rgb[0] + 150 * rgb[1] + 29 * rgb[2] + 128) >> 8;
      for (int channel = 0; channel < kInputChannels; ++channel) {
        const uint8_t reduced_chroma =
            static_cast<uint8_t>((luma + rgb[channel] + 1) >> 1);
        input->data.int8[output_offset + channel] =
            Quantize(kGamma12Lut[reduced_chroma], scale, zero);
      }
    }
  }
  return true;
}

bool ConfigureCameraSensor() {
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    ESP_LOGE(kTag, "Camera sensor handle is unavailable");
    return false;
  }
  if (sensor->id.PID == OV3660_PID) {
    // Espressif's reference setup notes that the OV3660 starts vertically
    // flipped and over-saturated. Correct it before collecting model frames.
    const int vflip_result = sensor->set_vflip(sensor, 1);
    const int brightness_result = sensor->set_brightness(sensor, 1);
    const int saturation_result = sensor->set_saturation(sensor, -2);
    if (vflip_result != 0 || brightness_result != 0 || saturation_result != 0) {
      ESP_LOGE(kTag, "OV3660 correction failed: vflip=%d brightness=%d saturation=%d",
               vflip_result, brightness_result, saturation_result);
      return false;
    }
    ESP_LOGI(kTag, "OV3660 corrected: vflip=1 brightness=1 saturation=-2");

    // Keep the sensor's automatic controls enabled and give AGC enough range
    // for the low-light OV3660 module used on this board.
    const int exposure_result = sensor->set_exposure_ctrl(sensor, 1);
    const int aec2_result = sensor->set_aec2(sensor, 1);
    const int ae_level_result = sensor->set_ae_level(sensor, 1);
    const int gain_result = sensor->set_gain_ctrl(sensor, 1);
    const int ceiling_result = sensor->set_gainceiling(sensor, GAINCEILING_64X);
    const int white_balance_result = sensor->set_whitebal(sensor, 1);
    const int awb_gain_result = sensor->set_awb_gain(sensor, 1);
    if (exposure_result != 0 || aec2_result != 0 || ae_level_result != 0 ||
        gain_result != 0 || ceiling_result != 0 || white_balance_result != 0 ||
        awb_gain_result != 0) {
      ESP_LOGE(kTag,
               "OV3660 auto-control setup failed: aec=%d aec2=%d ae=%d agc=%d ceiling=%d "
               "awb=%d awb_gain=%d",
               exposure_result, aec2_result, ae_level_result, gain_result, ceiling_result,
               white_balance_result, awb_gain_result);
      return false;
    }
    ESP_LOGI(kTag, "OV3660 low-light controls: AEC/AGC/AWB on, AE=+1, gain ceiling=64x");
  } else {
    ESP_LOGI(kTag, "Camera sensor PID=0x%04x; no OV3660 correction needed", sensor->id.PID);
  }

  // Initialise JPEG at a larger size so the driver reserves a sufficiently
  // large variable-length framebuffer, then run inference at 160x120.
  const int framesize_result = sensor->set_framesize(sensor, FRAMESIZE_QQVGA);
  if (framesize_result != 0) {
    ESP_LOGE(kTag, "Failed to set inference capture size: %d", framesize_result);
    return false;
  }
  ESP_LOGI(kTag, "Inference capture size set to 160x120");
  const int colorbar_result = sensor->set_colorbar(sensor, 1);
  if (colorbar_result != 0) {
    ESP_LOGE(kTag, "Failed to enable one-frame camera color-bar diagnostic: %d",
             colorbar_result);
    return false;
  }
  // The driver can already have a live-lens frame queued when the register is
  // changed. Discard transition frames so the first model input is a complete
  // sensor-generated color bar.
  for (int index = 0; index < 3; ++index) {
    camera_fb_t* transition = esp_camera_fb_get();
    if (transition == nullptr) {
      ESP_LOGE(kTag, "Failed to discard color-bar transition frame %d", index);
      return false;
    }
    esp_camera_fb_return(transition);
  }
  ESP_LOGI(kTag, "Camera color-bar stabilized for the first diagnostic frame");
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

void BenchmarkCopyBandwidth(const char* name, uint32_t caps, size_t bytes,
                            int iterations) {
  auto* source = static_cast<uint8_t*>(heap_caps_malloc(bytes, caps));
  auto* destination = static_cast<uint8_t*>(heap_caps_malloc(bytes, caps));
  if (source == nullptr || destination == nullptr) {
    ESP_LOGW(kTag, "%s copy benchmark skipped: allocation failed", name);
    heap_caps_free(source);
    heap_caps_free(destination);
    return;
  }
  std::memset(source, 0xa5, bytes);
  std::memset(destination, 0, bytes);
  uint8_t checksum = 0;
  const int64_t started_us = esp_timer_get_time();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::memcpy(destination, source, bytes);
    checksum ^= destination[(iteration * 97) % bytes];
  }
  const int64_t elapsed_us = esp_timer_get_time() - started_us;
  const float mib_per_second =
      (static_cast<float>(bytes) * iterations * 1000000.0f) /
      (static_cast<float>(elapsed_us) * 1024.0f * 1024.0f);
  ESP_LOGI(kTag, "%s memcpy: %.2f MiB/s (%u bytes x %d, %.1fms, check=%u)",
           name, mib_per_second, static_cast<unsigned>(bytes), iterations,
           elapsed_us / 1000.0f, checksum);
  heap_caps_free(source);
  heap_caps_free(destination);
}

void BenchmarkMemoryBandwidth() {
  BenchmarkCopyBandwidth("internal SRAM", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT,
                         32 * 1024, 256);
  BenchmarkCopyBandwidth("mapped PSRAM", MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
                         64 * 1024, 128);
}

bool BenchmarkCameraCapture(int frames) {
  size_t total_jpeg_bytes = 0;
  size_t minimum_jpeg_bytes = static_cast<size_t>(-1);
  size_t maximum_jpeg_bytes = 0;
  const int64_t started_us = esp_timer_get_time();
  for (int index = 0; index < frames; ++index) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ESP_LOGW(kTag, "Camera-only benchmark stopped at frame %d", index);
      return false;
    }
    total_jpeg_bytes += frame->len;
    minimum_jpeg_bytes = std::min(minimum_jpeg_bytes, frame->len);
    maximum_jpeg_bytes = std::max(maximum_jpeg_bytes, frame->len);
    esp_camera_fb_return(frame);
  }
  const int64_t elapsed_us = esp_timer_get_time() - started_us;
  const float frames_per_second = frames * 1000000.0f / elapsed_us;
  const float mean_jpeg_bytes = static_cast<float>(total_jpeg_bytes) / frames;
  ESP_LOGI(kTag,
           "Camera-only: %.2f fps (%d frames, %.1fms), JPEG mean=%.1fB range=[%u,%u] "
           "payload=%.1fkbit/s",
           frames_per_second, frames, elapsed_us / 1000.0f, mean_jpeg_bytes,
           static_cast<unsigned>(minimum_jpeg_bytes),
           static_cast<unsigned>(maximum_jpeg_bytes),
           mean_jpeg_bytes * frames_per_second * 8.0f / 1000.0f);
  return true;
}

bool ConfigureLed(gpio_num_t gpio, bool active_low) {
  gpio_config_t output{};
  output.pin_bit_mask = 1ULL << gpio;
  output.mode = GPIO_MODE_OUTPUT;
  output.pull_up_en = GPIO_PULLUP_DISABLE;
  output.pull_down_en = GPIO_PULLDOWN_DISABLE;
  output.intr_type = GPIO_INTR_DISABLE;
  if (gpio_config(&output) != ESP_OK) return false;
  return gpio_set_level(gpio, active_low ? 1 : 0) == ESP_OK;
}

void SetLed(gpio_num_t gpio, bool active_low, bool on) {
  gpio_set_level(gpio, active_low ? !on : on);
}

bool InitializeInferenceIndicators() {
  if (!ConfigureLed(kNonPersonLedGpio, kNonPersonLedActiveLow) ||
      !ConfigureLed(kFlashLedGpio, false)) {
    return false;
  }
  SetLed(kNonPersonLedGpio, kNonPersonLedActiveLow, true);
  SetLed(kFlashLedGpio, false, false);
  return true;
}

void SetInferenceIndicator(bool person_detected) {
  SetLed(kNonPersonLedGpio, kNonPersonLedActiveLow, !person_detected);
  SetLed(kFlashLedGpio, false, kFlashLedEnabled && person_detected);
}
}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "VWW startup: model=%u bytes threshold=%.2f debounce=%d/%d",
           g_vww_model_data_len, kPersonThreshold, kWakeFramesRequired, kDebounceWindowFrames);
  if (!esp_psram_is_initialized()) {
    ESP_LOGE(kTag, "PSRAM is not initialized; this build requires an ESP32-CAM with PSRAM");
    return;
  }
  if (!InitializeInferenceIndicators()) {
    ESP_LOGE(kTag, "Inference indicator GPIO initialization failed");
    return;
  }
  ESP_LOGI(kTag, "Indicators: RED=non-person; webpage BLUE=person; WHITE FLASH=%s",
           kFlashLedEnabled ? "enabled" : "disabled (forced low)");
  LogMemory("before allocations");
  BenchmarkMemoryBandwidth();

  // Direct RGB565 capture produced corrupted, nearly black rows on the tested
  // ESP32 + OV3660. Espressif recommends JPEG capture followed by RGB decode
  // when RGB pixels are needed on the original ESP32.
  camera_config_t camera = MakeBoardCameraConfig(PIXFORMAT_JPEG, FRAMESIZE_HD);
  const esp_err_t camera_result = esp_camera_init(&camera);
  if (camera_result != ESP_OK) {
    ESP_LOGE(kTag, "Camera init failed: 0x%x", camera_result);
    return;
  }
  if (!ConfigureCameraSensor()) return;

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
  auto* bgr888 = static_cast<uint8_t*>(
      heap_caps_malloc(kRgbScratchBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  auto* previous_input = static_cast<int8_t*>(
      heap_caps_malloc(kInputBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (arena == nullptr || bgr888 == nullptr || previous_input == nullptr) {
    ESP_LOGE(kTag, "PSRAM allocation failed: arena=%p bgr=%p previous=%p", arena, bgr888,
             previous_input);
    return;
  }

  // Static storage keeps the profiler's fixed accumulator off the small
  // ESP-IDF main-task stack.
  static DeviceOperatorProfiler operator_profiler;
  tflite::MicroInterpreter interpreter(model, resolver, arena, kTensorArenaBytes,
                                       nullptr, &operator_profiler);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed; increase kTensorArenaBytes");
    return;
  }
  TfLiteTensor* input = interpreter.input(0);
  const TfLiteTensor* output = interpreter.output(0);
  if (!ValidateTensorContract(input, output)) return;
  if (!InitializeWebDashboard()) {
    ESP_LOGE(kTag, "Web dashboard initialization failed");
    return;
  }
  PublishWebStatus(DashboardState::kStarting, 0.0f, 0.0f, 0.0f);
  LogMemory("ready");

  WakeDebouncer debounce;
  bool previous_wake = false;
  bool has_previous_input = false;
  bool camera_self_test_pending = true;
  uint32_t frame_number = 0;
  int64_t previous_loop_started_us = 0;
  while (true) {
    const int64_t loop_started_us = esp_timer_get_time();
    const int64_t frame_period_us =
        previous_loop_started_us == 0 ? -1 : loop_started_us - previous_loop_started_us;
    previous_loop_started_us = loop_started_us;
    camera_fb_t* frame = esp_camera_fb_get();
    const int64_t capture_us = esp_timer_get_time() - loop_started_us;
    if (frame == nullptr) {
      ESP_LOGW(kTag, "Camera capture failed");
      PublishWebStatus(DashboardState::kCameraError, 0.0f, 0.0f, 0.0f);
      vTaskDelay(pdMS_TO_TICKS(kFrameIntervalMs));
      continue;
    }
    const size_t jpeg_bytes = frame->len;
    const int64_t preprocessing_started_us = esp_timer_get_time();
    const bool converted = ResizeFrameToInt8(frame, bgr888, input);
    const int64_t preprocessing_us = esp_timer_get_time() - preprocessing_started_us;
    const int64_t publish_started_us = esp_timer_get_time();
    if (!camera_self_test_pending && frame->format == PIXFORMAT_JPEG) {
      StageWebFrame(frame->buf, frame->len);
    }
    const int64_t publish_us = esp_timer_get_time() - publish_started_us;
    esp_camera_fb_return(frame);
    if (!converted) {
      ESP_LOGE(kTag, "JPEG-to-RGB conversion failed");
      CommitWebFrameStatus(DashboardState::kCameraError, 0.0f, 0.0f, 0.0f,
                           false, 0, 0);
      continue;
    }

    if (camera_self_test_pending) {
      const InputStats self_test = SummarizeInput(input, nullptr, false);
      const float colorbar_mean =
          (self_test.mean[0] + self_test.mean[1] + self_test.mean[2]) / kInputChannels;
      ESP_LOGI(kTag,
               "Camera self-test: rgb_mean=[%.1f,%.1f,%.1f] range=[%u,%u] hash=%08lx",
               self_test.mean[0], self_test.mean[1], self_test.mean[2], self_test.minimum,
               self_test.maximum, static_cast<unsigned long>(self_test.checksum));
      LogAsciiPreview(input);

      sensor_t* sensor = esp_camera_sensor_get();
      if (sensor == nullptr || sensor->set_colorbar(sensor, 0) != 0) {
        ESP_LOGE(kTag, "Camera self-test failed: unable to disable sensor color-bar");
        PublishWebStatus(DashboardState::kCameraError, 0.0f, colorbar_mean, 0.0f);
        return;
      }
      if (colorbar_mean < kMinimumColorbarMean) {
        ESP_LOGE(kTag,
                 "CAMERA SELF-TEST FAILED: color-bar mean %.1f < %.1f; check/reseat the "
                 "camera ribbon, 5V power, or replace the OV3660 module",
                 colorbar_mean, kMinimumColorbarMean);
        SetInferenceIndicator(false);
        PublishWebStatus(DashboardState::kCameraError, 0.0f, colorbar_mean, 0.0f);
        return;
      }
      ESP_LOGI(kTag, "Camera self-test passed (color-bar mean %.1f)", colorbar_mean);

      for (int index = 0; index < 3; ++index) {
        camera_fb_t* transition = esp_camera_fb_get();
        if (transition == nullptr) {
          ESP_LOGE(kTag, "Failed to discard live-lens transition frame %d", index);
          PublishWebStatus(DashboardState::kCameraError, 0.0f, colorbar_mean, 0.0f);
          return;
        }
        esp_camera_fb_return(transition);
      }
      if (!BenchmarkCameraCapture(20)) {
        PublishWebStatus(DashboardState::kCameraError, 0.0f, colorbar_mean, 0.0f);
        return;
      }
      camera_self_test_pending = false;
      PublishWebStatus(DashboardState::kStarting, 0.0f, colorbar_mean, 0.0f);
      continue;
    }

    const bool should_log = (frame_number++ % kLogEveryFrames) == 0;
    const InputStats live_stats =
        SummarizeInput(input, previous_input, has_previous_input);
    std::memcpy(previous_input, input->data.int8, kInputBytes);
    has_previous_input = true;
    if (should_log && frame_number == 1) LogAsciiPreview(input);
    const float live_mean =
        (live_stats.mean[0] + live_stats.mean[1] + live_stats.mean[2]) / kInputChannels;
    if (live_mean < kMinimumLiveMean) {
      debounce.Reset();
      SetInferenceIndicator(false);
      CommitWebFrameStatus(DashboardState::kTooDark, 0.0f, live_mean, 0.0f,
                           false, 0, 0);
      if (previous_wake) {
        ESP_LOGW(kTag, "WAKE=OFF because camera input became too dark");
        previous_wake = false;
      }
      if (should_log) {
        ESP_LOGW(kTag,
                 "CAMERA TOO DARK: mean=%.1f < %.1f; inference skipped (remove lens cover or "
                 "add front lighting)",
                 live_mean, kMinimumLiveMean);
        LogAsciiPreview(input);
      }
      vTaskDelay(pdMS_TO_TICKS(kFrameIntervalMs));
      continue;
    }

    operator_profiler.BeginInference();
    const int64_t started_us = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
      operator_profiler.EndInference();
      ESP_LOGE(kTag, "Inference failed");
      CommitWebFrameStatus(DashboardState::kCameraError, 0.0f, live_mean, 0.0f,
                           false, debounce.PositiveVotes(), debounce.SampleCount());
      continue;
    }
    const int64_t latency_us = esp_timer_get_time() - started_us;
    operator_profiler.EndInference();
    if (operator_profiler.ReadyToReport()) {
      operator_profiler.LogReport(interpreter.arena_used_bytes());
    }
    output = interpreter.output(0);
    const float probability =
        (static_cast<int>(output->data.int8[0]) - output->params.zero_point) * output->params.scale;
    const bool raw_person = probability >= kPersonThreshold;
    const bool activation_motion = live_stats.mean_absolute_delta >= kActivationMotionThreshold;
    const bool wake = debounce.Update(raw_person, activation_motion);
    SetInferenceIndicator(wake);
    CommitWebFrameStatus(wake ? DashboardState::kPerson : DashboardState::kNonPerson,
                         probability, live_mean, latency_us / 1000.0f, raw_person,
                         debounce.PositiveVotes(), debounce.SampleCount(),
                         live_stats.mean_absolute_delta, activation_motion);
    if (wake != previous_wake) {
      ESP_LOGW(kTag, "WAKE=%s probability=%.3f motion=%.1f", wake ? "ON" : "OFF",
               probability, live_stats.mean_absolute_delta);
      previous_wake = wake;
    }
    if (should_log) {
      const int64_t active_pipeline_us = esp_timer_get_time() - loop_started_us;
      ESP_LOGI(kTag,
               "p=%.3f wake=%d jpeg=%uB timing_ms[capture=%.1f preprocess=%.1f "
               "infer=%.1f publish=%.1f active=%.1f period=%.1f fps=%.2f] "
               "rgb_mean=[%.1f,%.1f,%.1f] range=[%u,%u] delta=%.1f hash=%08lx",
               probability, wake, static_cast<unsigned>(jpeg_bytes), capture_us / 1000.0f,
               preprocessing_us / 1000.0f, latency_us / 1000.0f, publish_us / 1000.0f,
               active_pipeline_us / 1000.0f, frame_period_us / 1000.0f,
               frame_period_us > 0 ? 1000000.0f / frame_period_us : 0.0f,
               live_stats.mean[0], live_stats.mean[1], live_stats.mean[2], live_stats.minimum,
               live_stats.maximum, live_stats.mean_absolute_delta,
               static_cast<unsigned long>(live_stats.checksum));
    }
    vTaskDelay(pdMS_TO_TICKS(kFrameIntervalMs));
  }
}
