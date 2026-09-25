idf_component_register(
  SRCS "app_main.cc"
  INCLUDE_DIRS "." "../include"
  REQUIRES esp-tflite-micro esp_hw_support esp_psram esp_system esp_timer spi_flash
)

target_compile_definitions(
  ${COMPONENT_LIB}
  PRIVATE
    VWW_INPUT_SIZE=${VWW_INPUT_SIZE}
    VWW_MODEL_VARIANT="${VWW_MODEL_VARIANT}"
    VWW_ARENA_BYTES=${VWW_ARENA_BYTES}
)
