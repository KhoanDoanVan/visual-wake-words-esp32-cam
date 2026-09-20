from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_uses_same_origin_frame_endpoint():
    page = (ROOT / "firmware/esp32_cam_vww/main/web/dist/index.html").read_text()
    assert "'/frame?sequence='" in page
    assert ":81/stream" not in page
    assert "requestFrame(data.frame_sequence)" in page


def test_frame_endpoint_is_registered_on_main_http_server():
    source = (ROOT / "firmware/esp32_cam_vww/main/web_server.cc").read_text()
    assert 'frame.uri = "/frame"' in source
    assert "frame.handler = FrameHandler" in source
    assert "heap_caps_malloc(frame_length" in source
    assert "requested_sequence != g_telemetry.frame_sequence" in source


def test_frame_and_inference_are_committed_together():
    source = (ROOT / "firmware/esp32_cam_vww/main/web_server.cc").read_text()
    stage_body = source.split("bool StageWebFrame", 1)[1].split("void PublishWebStatus", 1)[0]
    commit_body = source.split("void CommitWebFrameStatus", 1)[1]
    assert "++g_telemetry.frame_sequence" not in stage_body
    assert "std::swap(g_latest_jpeg, g_pending_jpeg)" in commit_body
    assert "++g_telemetry.frame_sequence" in commit_body


def test_dashboard_exposes_raw_and_temporal_decisions():
    page = (ROOT / "firmware/esp32_cam_vww/main/web/dist/index.html").read_text()
    assert "data.raw_person" in page
    assert "data.positive_votes" in page
    assert "setInterval(update, 400)" in page
    assert "data.motion" in page
    assert "data.activation_allowed" in page


def test_temporal_vote_is_responsive_at_device_frame_rate():
    config = (ROOT / "firmware/esp32_cam_vww/main/app_config.h").read_text()
    app = (ROOT / "firmware/esp32_cam_vww/main/app_main.cc").read_text()
    assert "kDebounceWindowFrames = 3" in config
    assert "kWakeFramesRequired = 2" in config
    assert "samples_ == history_.size()" not in app


def test_real_camera_preprocessing_is_fixed_point_and_buffer_free():
    config = (ROOT / "firmware/esp32_cam_vww/main/app_config.h").read_text()
    app = (ROOT / "firmware/esp32_cam_vww/main/app_main.cc").read_text()
    resize_body = app.split("bool ResizeFrameToInt8", 1)[1].split(
        "bool ConfigureCameraSensor", 1
    )[0]
    assert "kPersonThreshold = 0.47f" in config
    assert "kGamma12Lut" in resize_body
    assert "77 * rgb[0] + 150 * rgb[1] + 29 * rgb[2]" in resize_body
    assert "(luma + rgb[channel] + 1) >> 1" in resize_body
    assert "pow" not in resize_body


def test_static_background_cannot_activate_visual_wake_word():
    config = (ROOT / "firmware/esp32_cam_vww/main/app_config.h").read_text()
    app = (ROOT / "firmware/esp32_cam_vww/main/app_main.cc").read_text()
    assert "kActivationMotionThreshold = 2.0f" in config
    assert "active_ ? detected : (detected && activation_allowed)" in app
    assert "debounce.Update(raw_person, activation_motion)" in app
    assert "std::memcpy(previous_input, input->data.int8, kInputBytes)" in app


def test_flash_feature_follows_the_stabilized_person_state_when_enabled():
    config = (ROOT / "firmware/esp32_cam_vww/main/app_config.h").read_text()
    app = (ROOT / "firmware/esp32_cam_vww/main/app_main.cc").read_text()
    initialize_body = app.split("bool InitializeInferenceIndicators", 1)[1].split(
        "void SetInferenceIndicator", 1
    )[0]
    indicator_body = app.split("void SetInferenceIndicator", 1)[1].split(
        "}  // namespace", 1
    )[0]
    inference_body = app.split("const bool raw_person", 1)[1].split(
        "CommitWebFrameStatus", 1
    )[0]
    assert "kFlashLedEnabled = true" in config
    assert "SetLed(kFlashLedGpio, false, false)" in initialize_body
    assert (
        "SetLed(kFlashLedGpio, false, kFlashLedEnabled && person_detected)"
        in indicator_body
    )
    assert "const bool wake = debounce.Update(raw_person, activation_motion)" in inference_body
    assert "SetInferenceIndicator(wake)" in inference_body
