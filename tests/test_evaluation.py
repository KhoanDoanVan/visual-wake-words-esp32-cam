import numpy as np

from vww_esp32.evaluation import classification_metrics, select_threshold


def test_metrics_and_threshold_selection():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    selected = select_threshold(labels, probabilities, minimum_recall=1.0)
    metrics = classification_metrics(labels, probabilities, selected["threshold"])
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
