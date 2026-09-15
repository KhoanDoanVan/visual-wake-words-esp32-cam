#pragma once

#include "driver/gpio.h"

// Threshold calibrated after the real-camera transform in app_main.cc. On the
// supplied device recording, non-transition person frames scored 0.484-0.602
// while non-person frames scored at most 0.434. Keep the transform and this
// threshold together; neither is compatible with the old raw-camera 0.27
// threshold.
constexpr float kPersonThreshold = 0.44f;

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

// GPIO33 drives the small red LED (active-low). GPIO4 drives the large white
// camera flash; keep it low at all times and show person detection in blue on
// the web dashboard instead.
constexpr gpio_num_t kNonPersonLedGpio = GPIO_NUM_33;
constexpr bool kNonPersonLedActiveLow = true;
constexpr gpio_num_t kFlashLedGpio = GPIO_NUM_4;
