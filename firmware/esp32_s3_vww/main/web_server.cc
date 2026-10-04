#include "web_server.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "app_config.h"

namespace {
constexpr char kTag[] = "vww_web";
constexpr char kAccessPointSsid[] = "VWW-Camera";
constexpr char kAccessPointPassword[] = "visualwake";
constexpr size_t kLatestJpegCapacity = 184320;

extern const uint8_t kWebPageStart[] asm("_binary_index_html_start");
extern const uint8_t kWebPageEnd[] asm("_binary_index_html_end");

struct DashboardTelemetry {
  DashboardState state = DashboardState::kStarting;
  float probability = 0.0f;
  float brightness = 0.0f;
  float inference_ms = 0.0f;
  bool raw_person = false;
  bool activation_allowed = false;
  float motion = -1.0f;
  int positive_votes = 0;
  int vote_samples = 0;
  uint32_t frame_sequence = 0;
  int64_t updated_ms = 0;
};

SemaphoreHandle_t g_dashboard_mutex = nullptr;
uint8_t* g_latest_jpeg = nullptr;
size_t g_latest_jpeg_length = 0;
uint8_t* g_pending_jpeg = nullptr;
size_t g_pending_jpeg_length = 0;
DashboardTelemetry g_telemetry;
httpd_handle_t g_http_server = nullptr;
httpd_handle_t g_stream_server = nullptr;

const char* StateName(DashboardState state) {
  switch (state) {
    case DashboardState::kPerson:
      return "person";
    case DashboardState::kNonPerson:
      return "non_person";
    case DashboardState::kTooDark:
      return "too_dark";
    case DashboardState::kCameraError:
      return "camera_error";
    case DashboardState::kStarting:
    default:
      return "starting";
  }
}

esp_err_t RootHandler(httpd_req_t* request) {
  httpd_resp_set_type(request, "text/html; charset=utf-8");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, reinterpret_cast<const char*>(kWebPageStart),
                         kWebPageEnd - kWebPageStart);
}

esp_err_t StatusHandler(httpd_req_t* request) {
  DashboardTelemetry telemetry;
  size_t jpeg_length = 0;
  if (xSemaphoreTake(g_dashboard_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    telemetry = g_telemetry;
    jpeg_length = g_latest_jpeg_length;
    xSemaphoreGive(g_dashboard_mutex);
  }

  char json[512];
  const int length = std::snprintf(
      json, sizeof(json),
      "{\"state\":\"%s\",\"probability\":%.4f,\"brightness\":%.1f,"
      "\"inference_ms\":%.1f,\"threshold\":%.2f,\"frame_sequence\":%lu,"
      "\"jpeg_bytes\":%u,\"raw_person\":%s,\"positive_votes\":%d,"
      "\"vote_samples\":%d,\"vote_window\":%d,\"votes_required\":%d,"
      "\"motion\":%.1f,\"motion_threshold\":%.1f,\"activation_allowed\":%s,"
      "\"updated_ms\":%lld}",
      StateName(telemetry.state), telemetry.probability, telemetry.brightness,
      telemetry.inference_ms, kPersonThreshold,
      static_cast<unsigned long>(telemetry.frame_sequence),
      static_cast<unsigned>(jpeg_length), telemetry.raw_person ? "true" : "false",
      telemetry.positive_votes, telemetry.vote_samples, kDebounceWindowFrames,
      kWakeFramesRequired, telemetry.motion, kActivationMotionThreshold,
      telemetry.activation_allowed ? "true" : "false",
      static_cast<long long>(telemetry.updated_ms));
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, json, std::max(0, length));
}

esp_err_t FrameHandler(httpd_req_t* request) {
  uint32_t requested_sequence = 0;
  bool has_requested_sequence = false;
  const size_t query_length = httpd_req_get_url_query_len(request);
  if (query_length > 0 && query_length < 96) {
    char query[96]{};
    char sequence[16]{};
    if (httpd_req_get_url_query_str(request, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, "sequence", sequence, sizeof(sequence)) == ESP_OK) {
      requested_sequence = static_cast<uint32_t>(std::strtoul(sequence, nullptr, 10));
      has_requested_sequence = true;
    }
  }

  uint8_t* frame = nullptr;
  size_t frame_length = 0;
  bool sequence_changed = false;
  if (xSemaphoreTake(g_dashboard_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    sequence_changed = has_requested_sequence &&
                       requested_sequence != g_telemetry.frame_sequence;
    if (!sequence_changed) {
      frame_length = g_latest_jpeg_length;
      if (frame_length > 0) {
        frame = static_cast<uint8_t*>(
            heap_caps_malloc(frame_length, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (frame != nullptr) std::memcpy(frame, g_latest_jpeg, frame_length);
      }
    }
    xSemaphoreGive(g_dashboard_mutex);
  }

  if (sequence_changed) {
    httpd_resp_set_status(request, "409 Conflict");
    httpd_resp_set_type(request, "text/plain");
    return httpd_resp_sendstr(request, "Frame sequence changed; refresh status");
  }

  if (frame == nullptr || frame_length == 0) {
    heap_caps_free(frame);
    httpd_resp_set_status(request, "503 Service Unavailable");
    httpd_resp_set_type(request, "text/plain");
    httpd_resp_sendstr(request, "Camera frame unavailable");
    return frame == nullptr && frame_length > 0 ? ESP_ERR_NO_MEM : ESP_ERR_NOT_FOUND;
  }

  httpd_resp_set_type(request, "image/jpeg");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store, no-cache, must-revalidate");
  httpd_resp_set_hdr(request, "Pragma", "no-cache");
  const esp_err_t result = httpd_resp_send(
      request, reinterpret_cast<const char*>(frame), static_cast<ssize_t>(frame_length));
  heap_caps_free(frame);
  return result;
}

esp_err_t StreamHandler(httpd_req_t* request) {
  constexpr char kContentType[] = "multipart/x-mixed-replace;boundary=frame";
  constexpr char kBoundary[] = "\r\n--frame\r\n";
  auto* client_frame = static_cast<uint8_t*>(
      heap_caps_malloc(kLatestJpegCapacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (client_frame == nullptr) {
    httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR, "No stream buffer");
    return ESP_ERR_NO_MEM;
  }

  httpd_resp_set_type(request, kContentType);
  httpd_resp_set_hdr(request, "Cache-Control", "no-store, no-cache, must-revalidate");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  uint32_t previous_sequence = 0;
  esp_err_t result = ESP_OK;
  while (result == ESP_OK) {
    size_t frame_length = 0;
    uint32_t frame_sequence = previous_sequence;
    if (xSemaphoreTake(g_dashboard_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      frame_sequence = g_telemetry.frame_sequence;
      if (g_latest_jpeg_length > 0 && frame_sequence != previous_sequence) {
        frame_length = g_latest_jpeg_length;
        std::memcpy(client_frame, g_latest_jpeg, frame_length);
      }
      xSemaphoreGive(g_dashboard_mutex);
    }
    if (frame_length == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    char header[96];
    const int header_length = std::snprintf(
        header, sizeof(header), "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
        static_cast<unsigned>(frame_length));
    result = httpd_resp_send_chunk(request, kBoundary, sizeof(kBoundary) - 1);
    if (result == ESP_OK) result = httpd_resp_send_chunk(request, header, header_length);
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, reinterpret_cast<const char*>(client_frame),
                                     frame_length);
    }
    previous_sequence = frame_sequence;
  }

  heap_caps_free(client_frame);
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
      esp_event_loop_create_default() != ESP_OK || esp_netif_create_default_wifi_ap() == nullptr) {
    return false;
  }

  wifi_init_config_t wifi_init = WIFI_INIT_CONFIG_DEFAULT();
  if (esp_wifi_init(&wifi_init) != ESP_OK || esp_wifi_set_storage(WIFI_STORAGE_RAM) != ESP_OK ||
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
  if (esp_wifi_set_config(WIFI_IF_AP, &wifi) != ESP_OK || esp_wifi_start() != ESP_OK) return false;
  return true;
}

bool StartHttpServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.stack_size = 8192;
  config.max_uri_handlers = 3;
  config.lru_purge_enable = true;
  if (httpd_start(&g_http_server, &config) != ESP_OK) return false;

  httpd_uri_t root{};
  root.uri = "/";
  root.method = HTTP_GET;
  root.handler = RootHandler;
  httpd_uri_t status{};
  status.uri = "/status";
  status.method = HTTP_GET;
  status.handler = StatusHandler;
  httpd_uri_t frame{};
  frame.uri = "/frame";
  frame.method = HTTP_GET;
  frame.handler = FrameHandler;
  if (httpd_register_uri_handler(g_http_server, &root) != ESP_OK ||
      httpd_register_uri_handler(g_http_server, &status) != ESP_OK ||
      httpd_register_uri_handler(g_http_server, &frame) != ESP_OK) {
    httpd_stop(g_http_server);
    g_http_server = nullptr;
    return false;
  }

  // A streaming handler remains open, so it needs its own server task. Keeping
  // telemetry on port 80 allows /status to respond while MJPEG runs on port 81.
  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  ++stream_config.ctrl_port;
  stream_config.stack_size = 8192;
  stream_config.max_uri_handlers = 1;
  stream_config.lru_purge_enable = true;
  if (httpd_start(&g_stream_server, &stream_config) != ESP_OK) {
    httpd_stop(g_http_server);
    g_http_server = nullptr;
    return false;
  }
  httpd_uri_t stream{};
  stream.uri = "/stream";
  stream.method = HTTP_GET;
  stream.handler = StreamHandler;
  if (httpd_register_uri_handler(g_stream_server, &stream) != ESP_OK) {
    httpd_stop(g_stream_server);
    httpd_stop(g_http_server);
    g_stream_server = nullptr;
    g_http_server = nullptr;
    return false;
  }
  return true;
}
}  // namespace

bool InitializeWebDashboard() {
  g_dashboard_mutex = xSemaphoreCreateMutex();
  g_latest_jpeg = static_cast<uint8_t*>(
      heap_caps_malloc(kLatestJpegCapacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  g_pending_jpeg = static_cast<uint8_t*>(
      heap_caps_malloc(kLatestJpegCapacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_dashboard_mutex == nullptr || g_latest_jpeg == nullptr || g_pending_jpeg == nullptr) {
    ESP_LOGE(kTag, "Dashboard buffer allocation failed");
    return false;
  }
  if (!StartAccessPoint() || !StartHttpServer()) {
    ESP_LOGE(kTag, "Dashboard network startup failed");
    return false;
  }
  ESP_LOGI(kTag, "Dashboard ready: Wi-Fi '%s' password '%s', open http://192.168.4.1",
           kAccessPointSsid, kAccessPointPassword);
  return true;
}

bool StageWebFrame(const uint8_t* jpeg, size_t length) {
  if (jpeg == nullptr || length == 0 || length > kLatestJpegCapacity ||
      g_dashboard_mutex == nullptr) {
    return false;
  }
  if (xSemaphoreTake(g_dashboard_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    std::memcpy(g_pending_jpeg, jpeg, length);
    g_pending_jpeg_length = length;
    xSemaphoreGive(g_dashboard_mutex);
    return true;
  }
  return false;
}

void PublishWebStatus(DashboardState state, float probability, float brightness,
                      float inference_ms) {
  if (g_dashboard_mutex == nullptr) return;
  if (xSemaphoreTake(g_dashboard_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    g_telemetry.state = state;
    g_telemetry.probability = probability;
    g_telemetry.brightness = brightness;
    g_telemetry.inference_ms = inference_ms;
    if (state != DashboardState::kPerson && state != DashboardState::kNonPerson) {
      g_telemetry.raw_person = false;
      g_telemetry.positive_votes = 0;
      g_telemetry.vote_samples = 0;
    }
    g_telemetry.updated_ms = esp_timer_get_time() / 1000;
    xSemaphoreGive(g_dashboard_mutex);
  }
}

void CommitWebFrameStatus(DashboardState state, float probability, float brightness,
                          float inference_ms, bool raw_person, int positive_votes,
                          int vote_samples, float motion, bool activation_allowed) {
  if (g_dashboard_mutex == nullptr) return;
  if (xSemaphoreTake(g_dashboard_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    g_telemetry.state = state;
    g_telemetry.probability = probability;
    g_telemetry.brightness = brightness;
    g_telemetry.inference_ms = inference_ms;
    g_telemetry.raw_person = raw_person;
    g_telemetry.motion = motion;
    g_telemetry.activation_allowed = activation_allowed;
    g_telemetry.positive_votes = positive_votes;
    g_telemetry.vote_samples = vote_samples;
    if (g_pending_jpeg_length > 0) {
      std::swap(g_latest_jpeg, g_pending_jpeg);
      g_latest_jpeg_length = g_pending_jpeg_length;
      g_pending_jpeg_length = 0;
      ++g_telemetry.frame_sequence;
    }
    g_telemetry.updated_ms = esp_timer_get_time() / 1000;
    xSemaphoreGive(g_dashboard_mutex);
  }
}
