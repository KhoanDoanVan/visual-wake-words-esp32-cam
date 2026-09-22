#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_private/esp_clk.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr char kTag[] = "s3_capacity";
constexpr size_t kArenaBytes = 160 * 1024;
constexpr int kProfileWarmup = 2;
constexpr int kProfileSamples = 20;
constexpr int kLatencyWarmup = 5;
constexpr int kLatencySamples = 100;

class OperatorProfiler final : public tflite::MicroProfilerInterface {
 public:
  explicit OperatorProfiler(const char* placement) : placement_(placement) {}

  void BeginInference() {
    current_ = 0;
    collecting_ = completed_ >= kProfileWarmup && measured_ < kProfileSamples;
    if (collecting_) started_ = esp_timer_get_time();
  }
  uint32_t BeginEvent(const char* tag) override {
    const uint32_t index = current_++;
    if (!collecting_ || index >= kMaxOps) return index;
    if (tags_[index] == nullptr) tags_[index] = tag;
    op_started_[index] = esp_timer_get_time();
    return index;
  }
  void EndEvent(uint32_t index) override {
    if (!collecting_ || index >= kMaxOps) return;
    const uint32_t elapsed = static_cast<uint32_t>(esp_timer_get_time() - op_started_[index]);
    timings_[index].total += elapsed;
    timings_[index].minimum = std::min(timings_[index].minimum, elapsed);
    timings_[index].maximum = std::max(timings_[index].maximum, elapsed);
    ++timings_[index].count;
  }
  void EndInference() {
    if (collecting_) {
      const uint32_t elapsed = static_cast<uint32_t>(esp_timer_get_time() - started_);
      total_ += elapsed;
      minimum_ = std::min(minimum_, elapsed);
      maximum_ = std::max(maximum_, elapsed);
      op_count_ = std::max(op_count_, current_);
      ++measured_;
    }
    ++completed_;
  }
  bool Ready() const { return measured_ == kProfileSamples; }
  void Report(size_t arena_used) const {
    ESP_LOGI(kTag,
             "VWW_PROFILE,BEGIN,variant=fast_80_s3_%s,input=80x80x3,batch=1,samples=%u,"
             "warmup=%d,arena_used_bytes=%u",
             placement_, static_cast<unsigned>(measured_), kProfileWarmup,
             static_cast<unsigned>(arena_used));
    uint64_t operator_total = 0;
    for (uint32_t index = 0; index < std::min(op_count_, kMaxOps); ++index) {
      const Timing& timing = timings_[index];
      operator_total += timing.total;
      ESP_LOGI(kTag,
               "VWW_PROFILE,OP,index=%u,tag=%s,count=%u,total_us=%llu,mean_us=%.2f,"
               "min_us=%u,max_us=%u",
               static_cast<unsigned>(index),
               tags_[index] == nullptr ? "UNKNOWN" : tags_[index],
               static_cast<unsigned>(timing.count),
               static_cast<unsigned long long>(timing.total),
               timing.count ? static_cast<double>(timing.total) / timing.count : 0.0,
               static_cast<unsigned>(timing.minimum == UINT32_MAX ? 0 : timing.minimum),
               static_cast<unsigned>(timing.maximum));
    }
    ESP_LOGI(kTag,
             "VWW_PROFILE,TOTAL,count=%u,total_us=%llu,mean_us=%.2f,min_us=%u,max_us=%u,"
             "operator_sum_mean_us=%.2f,unprofiled_mean_us=%.2f",
             static_cast<unsigned>(measured_), static_cast<unsigned long long>(total_),
             static_cast<double>(total_) / measured_, static_cast<unsigned>(minimum_),
             static_cast<unsigned>(maximum_),
             static_cast<double>(operator_total) / measured_,
             static_cast<double>(total_ - operator_total) / measured_);
    ESP_LOGI(kTag, "VWW_PROFILE,END,placement=%s", placement_);
  }

 private:
  static constexpr uint32_t kMaxOps = 64;
  struct Timing {
    uint64_t total = 0;
    uint32_t minimum = UINT32_MAX;
    uint32_t maximum = 0;
    uint32_t count = 0;
  };
  const char* placement_;
  std::array<const char*, kMaxOps> tags_{};
  std::array<int64_t, kMaxOps> op_started_{};
  std::array<Timing, kMaxOps> timings_{};
  uint32_t current_ = 0;
  uint32_t op_count_ = 0;
  uint32_t completed_ = 0;
  uint32_t measured_ = 0;
  int64_t started_ = 0;
  uint64_t total_ = 0;
  uint32_t minimum_ = UINT32_MAX;
  uint32_t maximum_ = 0;
  bool collecting_ = false;
};

// Keep the fixed timing accumulators off the small ESP-IDF main-task stack.
OperatorProfiler g_psram_profiler("psram");
OperatorProfiler g_internal_profiler("internal");

void LogMemory(const char* stage, const char* region, uint32_t caps) {
  ESP_LOGI(kTag,
           "S3_CAPACITY,MEM,stage=%s,region=%s,total=%u,free=%u,largest=%u,minimum=%u",
           stage, region, static_cast<unsigned>(heap_caps_get_total_size(caps)),
           static_cast<unsigned>(heap_caps_get_free_size(caps)),
           static_cast<unsigned>(heap_caps_get_largest_free_block(caps)),
           static_cast<unsigned>(heap_caps_get_minimum_free_size(caps)));
}

void LogAllMemory(const char* stage) {
  LogMemory(stage, "internal", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  LogMemory(stage, "psram", MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  LogMemory(stage, "dma", MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
}

void BenchmarkCopy(const char* direction, uint32_t source_caps, uint32_t destination_caps,
                   size_t bytes, int iterations) {
  auto* source = static_cast<uint8_t*>(heap_caps_malloc(bytes, source_caps));
  auto* destination = static_cast<uint8_t*>(heap_caps_malloc(bytes, destination_caps));
  if (source == nullptr || destination == nullptr) {
    ESP_LOGW(kTag, "S3_CAPACITY,BANDWIDTH,direction=%s,status=allocation_failed,bytes=%u",
             direction, static_cast<unsigned>(bytes));
    heap_caps_free(source);
    heap_caps_free(destination);
    return;
  }
  for (size_t i = 0; i < bytes; ++i) source[i] = static_cast<uint8_t>(i * 31U + 7U);
  memcpy(destination, source, bytes);
  const int64_t started = esp_timer_get_time();
  for (int i = 0; i < iterations; ++i) memcpy(destination, source, bytes);
  const int64_t elapsed = esp_timer_get_time() - started;
  volatile uint8_t checksum = destination[bytes / 2];
  const double mib_per_second =
      (static_cast<double>(bytes) * iterations * 1000000.0) / (elapsed * 1024.0 * 1024.0);
  ESP_LOGI(kTag,
           "S3_CAPACITY,BANDWIDTH,direction=%s,bytes=%u,iterations=%d,elapsed_us=%lld,"
           "mib_s=%.3f,checksum=%u",
           direction, static_cast<unsigned>(bytes), iterations,
           static_cast<long long>(elapsed), mib_per_second, static_cast<unsigned>(checksum));
  heap_caps_free(source);
  heap_caps_free(destination);
}

bool ValidateContract(TfLiteTensor* input, const TfLiteTensor* output) {
  const bool valid = input != nullptr && output != nullptr && input->type == kTfLiteInt8 &&
                     output->type == kTfLiteInt8 && input->dims->size == 4 &&
                     input->dims->data[0] == 1 && input->dims->data[1] == 80 &&
                     input->dims->data[2] == 80 && input->dims->data[3] == 3 &&
                     output->bytes == 1;
  if (!valid) ESP_LOGE(kTag, "Fast80 tensor contract mismatch");
  return valid;
}

void FillDeterministicInput(TfLiteTensor* input) {
  for (size_t i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = static_cast<int8_t>((i * 37U + 17U) % 256U - 128);
  }
}

void ProfileModel(const tflite::Model* model, tflite::MicroOpResolver& resolver,
                  const char* placement, uint32_t caps, OperatorProfiler& profiler) {
  LogAllMemory((std::strcmp(placement, "internal") == 0) ? "before_internal_model" :
                                                        "before_psram_model");
  auto* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, kArenaBytes, caps));
  if (arena == nullptr) {
    ESP_LOGE(kTag, "S3_CAPACITY,MODEL,placement=%s,status=arena_allocation_failed", placement);
    return;
  }
  {
    tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaBytes, nullptr, &profiler);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
      ESP_LOGE(kTag, "S3_CAPACITY,MODEL,placement=%s,status=allocate_tensors_failed", placement);
      heap_caps_free(arena);
      return;
    }
    TfLiteTensor* input = interpreter.input(0);
    const TfLiteTensor* output = interpreter.output(0);
    if (!ValidateContract(input, output)) {
      heap_caps_free(arena);
      return;
    }
    FillDeterministicInput(input);
    ESP_LOGI(kTag,
             "S3_CAPACITY,MODEL,placement=%s,status=ready,model_bytes=%u,arena_reserved=%u,"
             "arena_used=%u,input_bytes=%u,output_bytes=%u,input_scale=%.9g,input_zero=%d,"
             "output_scale=%.9g,output_zero=%d",
             placement, g_vww_model_data_len, static_cast<unsigned>(kArenaBytes),
             static_cast<unsigned>(interpreter.arena_used_bytes()),
             static_cast<unsigned>(input->bytes), static_cast<unsigned>(output->bytes),
             input->params.scale, static_cast<int>(input->params.zero_point),
             output->params.scale, static_cast<int>(output->params.zero_point));
    for (int i = 0; i < kProfileWarmup + kProfileSamples; ++i) {
      profiler.BeginInference();
      if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(kTag, "S3_CAPACITY,MODEL,placement=%s,status=invoke_failed", placement);
        return;
      }
      profiler.EndInference();
    }
    if (profiler.Ready()) profiler.Report(interpreter.arena_used_bytes());

    for (int i = 0; i < kLatencyWarmup; ++i) interpreter.Invoke();
    std::array<uint32_t, kLatencySamples> samples{};
    uint64_t total = 0;
    for (int i = 0; i < kLatencySamples; ++i) {
      const int64_t started = esp_timer_get_time();
      interpreter.Invoke();
      samples[i] = static_cast<uint32_t>(esp_timer_get_time() - started);
      total += samples[i];
      // Keep the CPU0 idle task healthy during the long benchmark. The yield
      // is deliberately outside the timed interval.
      vTaskDelay(pdMS_TO_TICKS(1));
    }
    std::sort(samples.begin(), samples.end());
    const int8_t raw = output->data.int8[0];
    const float probability = (raw - output->params.zero_point) * output->params.scale;
    ESP_LOGI(kTag,
             "S3_CAPACITY,LATENCY,placement=%s,count=%d,mean_us=%.2f,min_us=%u,median_us=%u,"
             "p95_us=%u,max_us=%u,output_raw=%d,output_probability=%.9g",
             placement, kLatencySamples, static_cast<double>(total) / kLatencySamples,
             static_cast<unsigned>(samples.front()),
             static_cast<unsigned>(samples[kLatencySamples / 2]),
             static_cast<unsigned>(samples[94]), static_cast<unsigned>(samples.back()),
             static_cast<int>(raw), probability);
  }
  heap_caps_free(arena);
  LogAllMemory((std::strcmp(placement, "internal") == 0) ? "after_internal_model" :
                                                        "after_psram_model");
}
}  // namespace

extern "C" void app_main(void) {
  uint32_t flash_bytes = 0;
  esp_flash_get_size(nullptr, &flash_bytes);
  esp_chip_info_t chip{};
  esp_chip_info(&chip);
  ESP_LOGI(kTag, "S3_CAPACITY,BEGIN,format=1");
  ESP_LOGI(kTag,
           "S3_CAPACITY,CHIP,model=%d,revision=%u,cores=%u,features=0x%08x,cpu_hz=%d,"
           "idf=%s",
           static_cast<int>(chip.model), static_cast<unsigned>(chip.revision),
           static_cast<unsigned>(chip.cores), static_cast<unsigned>(chip.features),
           esp_clk_cpu_freq(), esp_get_idf_version());
  ESP_LOGI(kTag,
           "S3_CAPACITY,STORAGE,flash_bytes=%u,psram_bytes=%u,flash_mode=DIO,"
           "flash_mhz=80,psram_mode=OCTAL,psram_mhz=80",
           static_cast<unsigned>(flash_bytes), static_cast<unsigned>(esp_psram_get_size()));
  LogAllMemory("boot");

  BenchmarkCopy("internal_to_internal", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT,
                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT, 32 * 1024, 256);
  BenchmarkCopy("psram_to_psram", MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, 256 * 1024, 128);
  BenchmarkCopy("internal_to_psram", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, 32 * 1024, 256);
  BenchmarkCopy("psram_to_internal", MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT, 32 * 1024, 256);

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

  ProfileModel(model, resolver, "psram", MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
               g_psram_profiler);
  ProfileModel(model, resolver, "internal", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT,
               g_internal_profiler);
  LogAllMemory("complete");
  ESP_LOGI(kTag, "S3_CAPACITY,END");
  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
