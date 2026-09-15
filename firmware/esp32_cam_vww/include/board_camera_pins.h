#pragma once

// AI-Thinker ESP32-CAM pinout. Verified on the photographed OV3660 module;
// the common OV2640 module uses the same board-level camera pin map.
inline camera_config_t MakeBoardCameraConfig(pixformat_t format, framesize_t size) {
  camera_config_t config{};
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.pin_xclk = 0;
  config.pin_sccb_sda = 26;
  config.pin_sccb_scl = 27;
  config.pin_d7 = 35;
  config.pin_d6 = 34;
  config.pin_d5 = 39;
  config.pin_d4 = 36;
  config.pin_d3 = 21;
  config.pin_d2 = 19;
  config.pin_d1 = 18;
  config.pin_d0 = 5;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_pclk = 22;
  // 10 MHz selects the non-high-speed ESP32 camera sampling path and is more
  // tolerant of marginal OV3660 ribbon/power/data timing.
  config.xclk_freq_hz = 10000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = format;
  config.frame_size = size;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  return config;
}
