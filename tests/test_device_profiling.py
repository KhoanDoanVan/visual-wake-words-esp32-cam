from pathlib import Path

import pytest

from vww_esp32.device_profiling import parse_device_profile_text, tflite_operator_inventory


def test_parse_complete_device_profile():
    text = """
I (1) vww: VWW_PROFILE,BEGIN,variant=fast_80,input=80x80x3,batch=1,samples=2,warmup=1,arena_used_bytes=123
I (2) vww: VWW_PROFILE,OP,index=0,tag=MUL,count=2,total_us=20,mean_us=10.0,min_us=9,max_us=11
I (3) vww: VWW_PROFILE,TOTAL,count=2,total_us=24,mean_us=12.0,min_us=11,max_us=13,operator_sum_mean_us=10.0,unprofiled_mean_us=2.0
I (4) vww: VWW_PROFILE,END\x1b[0m
"""
    operators, summary = parse_device_profile_text(text)
    assert operators.loc[0, "tag"] == "MUL"
    assert operators.loc[0, "mean_us"] == 10.0
    assert summary["variant"] == "fast_80"
    assert summary["operators"] == 1


def test_parse_rejects_partial_profile():
    with pytest.raises(ValueError, match="complete"):
        parse_device_profile_text("VWW_PROFILE,BEGIN,variant=broken")


def test_tflite_inventory_matches_deployed_model():
    model = Path("artifacts/fast_80/models/vww_mobilenetv1_80_int8.tflite")
    if not model.exists():
        pytest.skip("optimized model artifact is unavailable")
    operators = tflite_operator_inventory(model)
    assert len(operators) == 26
    assert operators.estimated_macs.sum() == 3_993_536
    assert operators.live_activation_during_bytes.max() == 38_400
    assert operators.constant_bytes.sum() == 111_806
    assert operators.iloc[0].constant_bytes == 1
    assert operators.iloc[0].operator == "MUL"
    assert operators.iloc[-1].operator == "LOGISTIC"
