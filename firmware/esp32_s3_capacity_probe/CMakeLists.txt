cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_s3_capacity_probe)

if(NOT DEFINED VWW_INPUT_SIZE)
  set(VWW_INPUT_SIZE 80)
endif()
if(NOT DEFINED VWW_MODEL_VARIANT)
  set(VWW_MODEL_VARIANT fast_80)
endif()
if(NOT DEFINED VWW_ARENA_BYTES)
  set(VWW_ARENA_BYTES 163840)
endif()
