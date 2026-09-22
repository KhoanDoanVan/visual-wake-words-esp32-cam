#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_camera.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "nvs_flash.h"

namespace {
constexpr char kTag[] = "s3_max_stream";
constexpr char kAccessPointSsid[] = "S3-Camera-Max";
constexpr char kAccessPointPassword[] = "visualwake";
constexpr size_t kMaxJpegBytes = 640 * 1024;
constexpr size_t kNetworkChunkBytes = 16 * 1024;

// Physically validated ESP32-S3-EYE-style pin map for this board.
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

extern const uint8_t kWebPageStart[] asm("_binary_index_html_start");
extern const uint8_t kWebPageEnd[] asm("_binary_index_html_end");

struct StreamTelemetry {
  uint32_t sequence = 0;
  size_t jpeg_bytes = 0;
  uint16_t width = 0;
  uint16_t height = 0;
  float capture_ms = 0.0f;
  float fps = 0.0f;
  int64_t updated_ms = 0;
  bool camera_ok = false;
};

SemaphoreHandle_t g_frame_mutex = nullptr;
uint8_t* g_latest_jpeg = nullptr;
StreamTelemetry g_telemetry;
httpd_handle_t g_main_server = nullptr;
httpd_handle_t g_stream_server = nullptr;

camera_config_t MakeCameraConfig() {
  camera_config_t config{};
  config.pin_pwdn = -1;
  config.pin_reset = -1;
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
  config.frame_size = FRAMESIZE_QXGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.sccb_i2c_port = -1;
  return config;
}

bool InitializeCamera() {
  camera_config_t config = MakeCameraConfig();
  const esp_err_t result = esp_camera_init(&config);
  if (result != ESP_OK) {
    ESP_LOGE(kTag, "Camera initialization failed: 0x%x", result);
    return false;
  }
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr || sensor->id.PID != OV3660_PID) {
    ESP_LOGE(kTag, "Expected OV3660, found PID=0x%04x", sensor == nullptr ? 0 : sensor->id.PID);
    return false;
  }
  // Match Espressif's OV3660 orientation/color corrections. Automatic
  // exposure, gain, and white balance remain enabled by the sensor defaults.
  sensor->set_vflip(sensor, 1);
  sensor->set_brightness(sensor, 1);
  sensor->set_saturation(sensor, -2);
  ESP_LOGI(kTag, "Camera ready: OV3660 PID=0x%04x QXGA=2048x1536 JPEG quality=12",
           sensor->id.PID);
  return true;
}

void CaptureTask(void*) {
  int64_t previous_started_us = 0;
  while (true) {
    const int64_t started_us = esp_timer_get_time();
    camera_fb_t* frame = esp_camera_fb_get();
    const int64_t finished_us = esp_timer_get_time();
    if (frame == nullptr) {
      if (xSemaphoreTake(g_frame_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        g_telemetry.camera_ok = false;
        g_telemetry.updated_ms = finished_us / 1000;
        xSemaphoreGive(g_frame_mutex);
      }
      ESP_LOGW(kTag, "Camera frame capture failed");
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    if (frame->format == PIXFORMAT_JPEG && frame->len <= kMaxJpegBytes &&
        xSemaphoreTake(g_frame_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      std::memcpy(g_latest_jpeg, frame->buf, frame->len);
      g_telemetry.jpeg_bytes = frame->len;
      g_telemetry.width = frame->width;
      g_telemetry.height = frame->height;
      g_telemetry.capture_ms = (finished_us - started_us) / 1000.0f;
      if (previous_started_us > 0) {
        const float instantaneous_fps = 1000000.0f / (started_us - previous_started_us);
        g_telemetry.fps = g_telemetry.fps == 0.0f
                              ? instantaneous_fps
                              : g_telemetry.fps * 0.85f + instantaneous_fps * 0.15f;
      }
      g_telemetry.updated_ms = finished_us / 1000;
      g_telemetry.camera_ok = true;
      ++g_telemetry.sequence;
      xSemaphoreGive(g_frame_mutex);
    }
    previous_started_us = started_us;
    esp_camera_fb_return(frame);
  }
}

esp_err_t RootHandler(httpd_req_t* request) {
  httpd_resp_set_type(request, "text/html; charset=utf-8");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, reinterpret_cast<const char*>(kWebPageStart),
                         kWebPageEnd - kWebPageStart);
}

esp_err_t StatusHandler(httpd_req_t* request) {
  StreamTelemetry telemetry;
  if (xSemaphoreTake(g_frame_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    telemetry = g_telemetry;
    xSemaphoreGive(g_frame_mutex);
  }
  char json[512];
  const int length = std::snprintf(
      json, sizeof(json),
      "{\"camera_ok\":%s,\"sensor\":\"OV3660\",\"width\":%u,\"height\":%u,"
      "\"jpeg_bytes\":%u,\"capture_ms\":%.2f,\"fps\":%.3f,\"sequence\":%lu,"
      "\"updated_ms\":%lld,\"psram_free\":%u,\"internal_free\":%u,"
      "\"flash_led\":false}",
      telemetry.camera_ok ? "true" : "false", telemetry.width, telemetry.height,
      static_cast<unsigned>(telemetry.jpeg_bytes), telemetry.capture_ms, telemetry.fps,
      static_cast<unsigned long>(telemetry.sequence),
      static_cast<long long>(telemetry.updated_ms),
      static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
      static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(request, json, std::max(0, length));
}

bool CopyLatestFrame(uint8_t* destination, size_t* length, uint32_t* sequence) {
  bool copied = false;
  if (xSemaphoreTake(g_frame_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    if (g_telemetry.jpeg_bytes > 0 && g_telemetry.jpeg_bytes <= kMaxJpegBytes) {
      *length = g_telemetry.jpeg_bytes;
      *sequence = g_telemetry.sequence;
      std::memcpy(destination, g_latest_jpeg, *length);
      copied = true;
    }
    xSemaphoreGive(g_frame_mutex);
  }
  return copied;
}

esp_err_t CaptureHandler(httpd_req_t* request) {
  auto* frame = static_cast<uint8_t*>(
      heap_caps_malloc(kMaxJpegBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (frame == nullptr) return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                                                   "No snapshot buffer");
  size_t length = 0;
  uint32_t sequence = 0;
  if (!CopyLatestFrame(frame, &length, &sequence)) {
    heap_caps_free(frame);
    httpd_resp_set_status(request, "503 Service Unavailable");
    httpd_resp_set_type(request, "text/plain");
    return httpd_resp_sendstr(request, "Camera frame unavailable");
  }
  char disposition[80];
  std::snprintf(disposition, sizeof(disposition),
                "attachment; filename=ov3660_qxga_%lu.jpg",
                static_cast<unsigned long>(sequence));
  httpd_resp_set_type(request, "image/jpeg");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  httpd_resp_set_hdr(request, "Content-Disposition", disposition);
  const esp_err_t result = httpd_resp_send(
      request, reinterpret_cast<const char*>(frame), static_cast<ssize_t>(length));
  heap_caps_free(frame);
  return result;
}

esp_err_t StreamHandler(httpd_req_t* request) {
  constexpr char kContentType[] = "multipart/x-mixed-replace;boundary=frame";
  constexpr char kBoundary[] = "\r\n--frame\r\n";
  auto* frame = static_cast<uint8_t*>(
      heap_caps_malloc(kMaxJpegBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (frame == nullptr) return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                                                   "No stream buffer");

  httpd_resp_set_type(request, kContentType);
  httpd_resp_set_hdr(request, "Cache-Control", "no-store, no-cache, must-revalidate");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  uint32_t previous_sequence = 0;
  esp_err_t result = ESP_OK;
  while (result == ESP_OK) {
    size_t length = 0;
    uint32_t sequence = 0;
    if (!CopyLatestFrame(frame, &length, &sequence) || sequence == previous_sequence) {
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }
    char header[128];
    const int header_length = std::snprintf(
        header, sizeof(header),
        "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Sequence: %lu\r\n\r\n",
        static_cast<unsigned>(length), static_cast<unsigned long>(sequence));
    result = httpd_resp_send_chunk(request, kBoundary, sizeof(kBoundary) - 1);
    if (result == ESP_OK) result = httpd_resp_send_chunk(request, header, header_length);
    for (size_t offset = 0; result == ESP_OK && offset < length;
         offset += kNetworkChunkBytes) {
      const size_t chunk = std::min(kNetworkChunkBytes, length - offset);
      result = httpd_resp_send_chunk(request,
                                     reinterpret_cast<const char*>(frame + offset), chunk);
    }
    previous_sequence = sequence;
  }
  heap_caps_free(frame);
  httpd_resp_send_chunk(request, nullptr, 0);
  return result;
}

bool StartAccessPoint() {
  esp_err_t nvs_result = nvs_flash_init();
  if (nvs_result == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    if (nvs_flash_erase() != ESP_OK) return false;
    nvs_result = nvs_flash_init();
  }
  if (nvs_result != ESP_OK || esp_netif_init() != ESP_OK ||
      esp_event_loop_create_default() != ESP_OK ||
      esp_netif_create_default_wifi_ap() == nullptr) {
    return false;
  }
  wifi_init_config_t initialization = WIFI_INIT_CONFIG_DEFAULT();
  if (esp_wifi_init(&initialization) != ESP_OK ||
      esp_wifi_set_storage(WIFI_STORAGE_RAM) != ESP_OK ||
      esp_wifi_set_mode(WIFI_MODE_AP) != ESP_OK) {
    return false;
  }
  wifi_config_t wifi{};
  std::memcpy(wifi.ap.ssid, kAccessPointSsid, sizeof(kAccessPointSsid));
  std::memcpy(wifi.ap.password, kAccessPointPassword, sizeof(kAccessPointPassword));
  wifi.ap.ssid_len = sizeof(kAccessPointSsid) - 1;
  wifi.ap.channel = 1;
  wifi.ap.max_connection = 2;
  wifi.ap.authmode = WIFI_AUTH_WPA2_PSK;
  return esp_wifi_set_config(WIFI_IF_AP, &wifi) == ESP_OK && esp_wifi_start() == ESP_OK;
}

bool StartWebServers() {
  httpd_config_t main_config = HTTPD_DEFAULT_CONFIG();
  main_config.stack_size = 8192;
  main_config.max_uri_handlers = 3;
  main_config.lru_purge_enable = true;
  if (httpd_start(&g_main_server, &main_config) != ESP_OK) return false;

  httpd_uri_t root{};
  root.uri = "/";
  root.method = HTTP_GET;
  root.handler = RootHandler;
  httpd_uri_t status{};
  status.uri = "/status";
  status.method = HTTP_GET;
  status.handler = StatusHandler;
  httpd_uri_t capture{};
  capture.uri = "/capture";
  capture.method = HTTP_GET;
  capture.handler = CaptureHandler;
  if (httpd_register_uri_handler(g_main_server, &root) != ESP_OK ||
      httpd_register_uri_handler(g_main_server, &status) != ESP_OK ||
      httpd_register_uri_handler(g_main_server, &capture) != ESP_OK) {
    return false;
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  ++stream_config.ctrl_port;
  stream_config.stack_size = 8192;
  stream_config.max_uri_handlers = 1;
  stream_config.lru_purge_enable = true;
  if (httpd_start(&g_stream_server, &stream_config) != ESP_OK) return false;
  httpd_uri_t stream{};
  stream.uri = "/stream";
  stream.method = HTTP_GET;
  stream.handler = StreamHandler;
  return httpd_register_uri_handler(g_stream_server, &stream) == ESP_OK;
}
}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "Starting maximum-resolution OV3660 stream; flash LED is disabled");
  if (!esp_psram_is_initialized()) {
    ESP_LOGE(kTag, "8 MB PSRAM is required");
    return;
  }
  g_frame_mutex = xSemaphoreCreateMutex();
  g_latest_jpeg = static_cast<uint8_t*>(
      heap_caps_malloc(kMaxJpegBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_frame_mutex == nullptr || g_latest_jpeg == nullptr) {
    ESP_LOGE(kTag, "Frame storage allocation failed");
    return;
  }
  if (!InitializeCamera()) return;
  if (xTaskCreatePinnedToCore(CaptureTask, "max_camera_capture", 4096, nullptr, 5, nullptr, 1) !=
      pdPASS) {
    ESP_LOGE(kTag, "Capture task creation failed");
    return;
  }
  if (!StartAccessPoint() || !StartWebServers()) {
    ESP_LOGE(kTag, "Wi-Fi or web server startup failed");
    return;
  }
  ESP_LOGI(kTag, "STREAM_READY,ssid=%s,password=%s,url=http://192.168.4.1/",
           kAccessPointSsid, kAccessPointPassword);
}
