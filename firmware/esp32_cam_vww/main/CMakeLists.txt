idf_component_register(
  SRCS "app_main.cc"
  INCLUDE_DIRS "." "../include"
  REQUIRES esp32-camera esp-tflite-micro esp_driver_gpio esp_psram esp_timer
)
