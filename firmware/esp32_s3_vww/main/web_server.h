#pragma once

#include <cstddef>
#include <cstdint>

enum class DashboardState {
  kStarting,
  kPerson,
  kNonPerson,
  kTooDark,
  kCameraError,
};

bool InitializeWebDashboard();
bool StageWebFrame(const uint8_t* jpeg, size_t length);
void PublishWebStatus(DashboardState state, float probability, float brightness,
                      float inference_ms);
void CommitWebFrameStatus(DashboardState state, float probability, float brightness,
                          float inference_ms, bool raw_person, int positive_votes,
                          int vote_samples, float motion = -1.0f,
                          bool activation_allowed = false);
