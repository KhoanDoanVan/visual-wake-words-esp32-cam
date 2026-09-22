#include <cstdint>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {
constexpr char kTag[] = "s3_camera_probe";

// Common ESP32-S3-WROOM N16R8 camera-board mapping. It is also the mapping
// used by Espressif's ESP32-S3-EYE camera definition. No illumination LED is
// configured or touched by this diagnostic.
constexpr int kPinPwdn = -1;
constexpr int kPinReset = -1;
constexpr int kPinXclk = 15;
constexpr int kPinSiod = 4;
constexpr int kPinSioc = 5;
constexpr int kPinD0 = 11;
constexpr int kPinD1 = 9;
constexpr int kPinD2 = 8;
constexpr int kPinD3 = 10;
constexpr int kPinD4 = 12;
constexpr int kPinD5 = 18;
constexpr int kPinD6 = 17;
constexpr int kPinD7 = 16;
constexpr int kPinVsync = 6;
constexpr int kPinHref = 7;
constexpr int kPinPclk = 13;

struct FrameSizeCase {
  framesize_t value;
  const char* name;
};

constexpr FrameSizeCase kFrameSizes[] = {
    {FRAMESIZE_QQVGA, "QQVGA"},
    {FRAMESIZE_QCIF, "QCIF"},     {FRAMESIZE_HQVGA, "HQVGA"},
    {FRAMESIZE_240X240, "240X240"}, {FRAMESIZE_QVGA, "QVGA"},
    {FRAMESIZE_CIF, "CIF"},       {FRAMESIZE_HVGA, "HVGA"},
    {FRAMESIZE_VGA, "VGA"},       {FRAMESIZE_SVGA, "SVGA"},
    {FRAMESIZE_XGA, "XGA"},       {FRAMESIZE_HD, "HD"},
    {FRAMESIZE_SXGA, "SXGA"},     {FRAMESIZE_UXGA, "UXGA"},
    {FRAMESIZE_FHD, "FHD"},       {FRAMESIZE_P_HD, "P_HD"},
    {FRAMESIZE_P_3MP, "P_3MP"},   {FRAMESIZE_QXGA, "QXGA"},
};

camera_config_t MakeCameraConfig() {
  camera_config_t config{};
  config.pin_pwdn = kPinPwdn;
  config.pin_reset = kPinReset;
  config.pin_xclk = kPinXclk;
  config.pin_sccb_sda = kPinSiod;
  config.pin_sccb_scl = kPinSioc;
  config.pin_d0 = kPinD0;
  config.pin_d1 = kPinD1;
  config.pin_d2 = kPinD2;
  config.pin_d3 = kPinD3;
  config.pin_d4 = kPinD4;
  config.pin_d5 = kPinD5;
  config.pin_d6 = kPinD6;
  config.pin_d7 = kPinD7;
  config.pin_vsync = kPinVsync;
  config.pin_href = kPinHref;
  config.pin_pclk = kPinPclk;
  config.xclk_freq_hz = 20000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = PIXFORMAT_JPEG;
  // Allocate a framebuffer large enough for the OV3660's maximum mode. The
  // driver does not grow the framebuffer when set_framesize() is called.
  config.frame_size = FRAMESIZE_QXGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.sccb_i2c_port = -1;
  return config;
}

void DiscardFrames(int count) {
  for (int i = 0; i < count; ++i) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) return;
    esp_camera_fb_return(frame);
  }
}

void ProbeFrameSizes(sensor_t* sensor) {
  uint32_t maximum_pixels = 0;
  uint16_t maximum_width = 0;
  uint16_t maximum_height = 0;
  const char* maximum_name = "NONE";

  ESP_LOGI(kTag, "CAMERA_PROBE,RESOLUTION_BEGIN");
  for (const FrameSizeCase& candidate : kFrameSizes) {
    const int set_result = sensor->set_framesize(sensor, candidate.value);
    if (set_result != 0) {
      ESP_LOGI(kTag, "CAMERA_PROBE,RESOLUTION,name=%s,set_result=%d,status=unsupported",
               candidate.name, set_result);
      continue;
    }

    vTaskDelay(pdMS_TO_TICKS(250));
    DiscardFrames(2);
    uint64_t bytes_total = 0;
    uint64_t capture_total_us = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    int successful = 0;
    for (int sample = 0; sample < 3; ++sample) {
      const int64_t started = esp_timer_get_time();
      camera_fb_t* frame = esp_camera_fb_get();
      const uint32_t elapsed = static_cast<uint32_t>(esp_timer_get_time() - started);
      if (frame == nullptr) break;
      width = frame->width;
      height = frame->height;
      bytes_total += frame->len;
      capture_total_us += elapsed;
      ++successful;
      esp_camera_fb_return(frame);
    }

    if (successful == 3) {
      const uint32_t pixels = static_cast<uint32_t>(width) * height;
      if (pixels > maximum_pixels) {
        maximum_pixels = pixels;
        maximum_width = width;
        maximum_height = height;
        maximum_name = candidate.name;
      }
      ESP_LOGI(kTag,
               "CAMERA_PROBE,RESOLUTION,name=%s,width=%u,height=%u,status=ok,samples=%d,"
               "jpeg_mean_bytes=%llu,capture_mean_us=%llu",
               candidate.name, width, height, successful,
               static_cast<unsigned long long>(bytes_total / successful),
               static_cast<unsigned long long>(capture_total_us / successful));
    } else {
      ESP_LOGI(kTag,
               "CAMERA_PROBE,RESOLUTION,name=%s,width=%u,height=%u,status=capture_failed,"
               "samples=%d",
               candidate.name, width, height, successful);
    }
  }
  ESP_LOGI(kTag,
           "CAMERA_PROBE,MAXIMUM,name=%s,width=%u,height=%u,pixels=%u,megapixels=%.4f",
           maximum_name, maximum_width, maximum_height,
           static_cast<unsigned>(maximum_pixels), maximum_pixels / 1000000.0);
  ESP_LOGI(kTag, "CAMERA_PROBE,RESOLUTION_END");
}
}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "CAMERA_PROBE,BEGIN,format=1,flash_led=disabled");
  ESP_LOGI(kTag,
           "CAMERA_PROBE,PINS,pwdn=%d,reset=%d,xclk=%d,siod=%d,sioc=%d,d0=%d,d1=%d,"
           "d2=%d,d3=%d,d4=%d,d5=%d,d6=%d,d7=%d,vsync=%d,href=%d,pclk=%d",
           kPinPwdn, kPinReset, kPinXclk, kPinSiod, kPinSioc, kPinD0, kPinD1, kPinD2,
           kPinD3, kPinD4, kPinD5, kPinD6, kPinD7, kPinVsync, kPinHref, kPinPclk);
  ESP_LOGI(kTag, "CAMERA_PROBE,MEMORY,psram_total=%u,psram_free=%u",
           static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)),
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

  camera_config_t config = MakeCameraConfig();
  const esp_err_t init_result = esp_camera_init(&config);
  if (init_result != ESP_OK) {
    ESP_LOGE(kTag, "CAMERA_PROBE,INIT,status=failed,error=0x%x", init_result);
    ESP_LOGE(kTag, "The board may use a different camera pin map or the ribbon is disconnected");
    while (true) vTaskDelay(pdMS_TO_TICKS(10000));
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    ESP_LOGE(kTag, "CAMERA_PROBE,SENSOR,status=missing");
    while (true) vTaskDelay(pdMS_TO_TICKS(10000));
  }
  ESP_LOGI(kTag,
           "CAMERA_PROBE,SENSOR,status=ok,pid=0x%04x,ver=0x%02x,mid_h=0x%02x,mid_l=0x%02x,"
           "address=0x%02x",
           sensor->id.PID, sensor->id.VER, sensor->id.MIDH, sensor->id.MIDL, sensor->slv_addr);
  ProbeFrameSizes(sensor);
  ESP_LOGI(kTag, "CAMERA_PROBE,END");
  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
