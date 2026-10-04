#pragma once

#include "driver/gpio.h"

// Frozen validation threshold for Student-120 hard-control INT8. A later
// labelled S3 camera capture can recalibrate this without retraining.
#ifndef VWW_PERSON_THRESHOLD
#define VWW_PERSON_THRESHOLD 0.275f
#endif
constexpr float kPersonThreshold = VWW_PERSON_THRESHOLD;

// Use a short 2-of-3 temporal vote. At the measured ~1.7 fps this confirms or
// clears a state in roughly 0.6-1.2 seconds while still rejecting an isolated
// frame. Do not wait for the entire window to fill during initial activation.
constexpr int kDebounceWindowFrames = 3;
constexpr int kWakeFramesRequired = 2;
constexpr int kReleaseFramesMaximum = 1;

// A dormant visual wake word must be accompanied by real scene motion. This
// prevents a static high-scoring background (for example the ceiling fixture
// in the supplied recording) from activating the wake state. Motion is the
// mean absolute difference in 8-bit model-input units from the previous frame.
// Once awake, the classifier alone maintains/releases the state so a person
// does not have to keep moving.
constexpr float kActivationMotionThreshold = 2.0f;

constexpr int kFrameIntervalMs = 100;
constexpr int kLogEveryFrames = 10;

// Collect operator timings for the first stable live inferences. The profiler
// emits machine-readable VWW_PROFILE records once, then becomes a near-no-op.
constexpr int kProfileWarmupInvocations = 2;
constexpr int kProfileMeasuredInvocations = 20;

// The official ESP32-S3-EYE BSP assigns its active-low status LED to GPIO3.
// This board has no verified white illumination flash, so "flash" means the
// inference indicator feature: GPIO3 lights only for stabilized person state.
// GPIO4/5 belong to the camera SCCB bus and must never be reused as ESP32-CAM
// flash pins on this hardware.
constexpr gpio_num_t kFlashLedGpio = GPIO_NUM_3;
constexpr bool kFlashLedActiveLow = true;
#ifndef VWW_FLASH_LED_ENABLED
#define VWW_FLASH_LED_ENABLED 1
#endif
constexpr bool kFlashLedEnabled = VWW_FLASH_LED_ENABLED != 0;
