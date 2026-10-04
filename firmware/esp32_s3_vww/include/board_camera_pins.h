#pragma once

// ESP32-S3-EYE pinout. The physical probe verified this exact map with the
// attached OV3660 through every JPEG frame size up to 2048x1536.
inline camera_config_t MakeBoardCameraConfig(pixformat_t format, framesize_t size) {
  camera_config_t config{};
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.pin_xclk = 15;
  config.pin_sccb_sda = 4;
  config.pin_sccb_scl = 5;
  config.pin_d0 = 11;
  config.pin_d1 = 9;
  config.pin_d2 = 8;
  config.pin_d3 = 10;
  config.pin_d4 = 12;
  config.pin_d5 = 18;
  config.pin_d6 = 17;
  config.pin_d7 = 16;
  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_pclk = 13;
  config.xclk_freq_hz = 20000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = format;
  config.frame_size = size;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.sccb_i2c_port = -1;
  return config;
}
