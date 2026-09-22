idf_component_register(
  SRCS "app_main.cc"
  INCLUDE_DIRS "." "../include"
  REQUIRES esp-tflite-micro esp_hw_support esp_psram esp_system esp_timer spi_flash
)
