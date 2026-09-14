from vww_esp32.exporting import write_c_header


def test_c_header_generation(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(bytes([0, 127, 255]))
    header = write_c_header(model, tmp_path / "model.h", "g_model")
    text = header.read_text()
    assert "alignas(16) const unsigned char g_model[]" in text
    assert "0x00, 0x7f, 0xff" in text
    assert "g_model_len = 3" in text
