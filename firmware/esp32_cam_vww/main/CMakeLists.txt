idf_component_register(
  SRCS "app_main.cc" "web_server.cc"
  INCLUDE_DIRS "." "../include"
  REQUIRES esp32-camera esp-tflite-micro esp_driver_gpio esp_http_server esp_psram esp_timer
           esp_wifi nvs_flash
  EMBED_TXTFILES "web/dist/index.html"
)

if(NOT DEFINED VWW_INPUT_SIZE)
  set(VWW_INPUT_SIZE 80)
endif()
if(NOT DEFINED VWW_MODEL_VARIANT)
  set(VWW_MODEL_VARIANT fast_80)
endif()
if(NOT DEFINED VWW_PERSON_THRESHOLD)
  set(VWW_PERSON_THRESHOLD 0.47f)
endif()
if(NOT DEFINED VWW_FLASH_LED_ENABLED)
  set(VWW_FLASH_LED_ENABLED 0)
endif()

target_compile_definitions(
  ${COMPONENT_LIB}
  PRIVATE
    VWW_INPUT_SIZE=${VWW_INPUT_SIZE}
    VWW_MODEL_VARIANT="${VWW_MODEL_VARIANT}"
    VWW_PERSON_THRESHOLD=${VWW_PERSON_THRESHOLD}
    VWW_FLASH_LED_ENABLED=${VWW_FLASH_LED_ENABLED}
)
