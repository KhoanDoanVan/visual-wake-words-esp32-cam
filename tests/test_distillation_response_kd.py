from pathlib import Path

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG = ROOT / "configs/distillation_student_120.yaml"
KD_CONFIG = ROOT / "configs/distillation_response_kd_120.yaml"
NOTEBOOK = (
    ROOT
    / "optimization/distillation/response-based distillation"
    / "02_response_kd_hinton_micronets.ipynb"
)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_response_kd_keeps_controlled_student_and_training_contract() -> None:
    control = read_yaml(CONTROL_CONFIG)
    response_kd = read_yaml(KD_CONFIG)

    for section in ("student", "preprocessing", "augmentation", "training"):
        assert response_kd[section] == control[section]

    for key, value in control["evaluation"].items():
        assert response_kd["evaluation"][key] == value

    assert response_kd["teacher"] == control["teacher"]
    assert response_kd["project"]["seed"] == control["project"]["seed"]


def test_response_kd_uses_the_registered_hinton_micronets_anchor() -> None:
    config = read_yaml(KD_CONFIG)
    method = config["distillation"]

    assert method["method"] == "binary_response_kd"
    assert method["temperature"] == 4.0
    assert method["teacher_loss_weight"] == 0.5
    assert method["multiply_soft_loss_by_temperature_squared"] is True
    assert method["teacher_inference_temperature"] == 1.0


def test_response_kd_notebook_is_complete_and_uses_cached_logits() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")

    assert len(notebook.cells) >= 60
    assert "class BinaryResponseDistiller" in code
    assert "teacher_logit" in code
    assert "tf.square(self.temperature)" in code
    assert "teacher_entropy" in code
    assert "load_model(TEACHER_MODEL_PATH" not in code
    assert "Hinton" in markdown
    assert "MicroNets" in markdown
    assert "Physical profiling" in markdown
