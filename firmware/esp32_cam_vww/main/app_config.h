#pragma once

#include "driver/gpio.h"

// Safer than the validation-selected 0.34 for an always-on camera: test FPR
// was ~20% at 0.50 versus ~43% at 0.34. Retune with device validation data.
constexpr float kPersonThreshold = 0.50f;

// Trigger after 4 positive frames among the latest 5 and release when the
// window falls to 1 positive frame. This prevents single-frame flicker.
constexpr int kDebounceWindowFrames = 5;
constexpr int kWakeFramesRequired = 4;
constexpr int kReleaseFramesMaximum = 1;

constexpr int kFrameIntervalMs = 100;
constexpr int kLogEveryFrames = 10;

// AI-Thinker ESP32-CAM red status LED. It is active-low.
constexpr gpio_num_t kWakeOutputGpio = GPIO_NUM_33;
constexpr bool kWakeOutputActiveLow = true;
