idf_component_register(
  SRCS "app_main.cc"
  INCLUDE_DIRS "."
  REQUIRES esp32-camera esp_driver_gpio esp_http_server esp_psram esp_timer esp_wifi nvs_flash
  EMBED_TXTFILES "web/index.html"
)
