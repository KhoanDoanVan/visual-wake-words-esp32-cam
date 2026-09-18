# Unstructured magnitude pruning report — revised

## Method contract

- Fast-80 is the frozen pruning baseline.
- Global one-shot is retained as the negative control.
- Paper-inspired branch uses 5 prune/retrain rounds, sensitivity-adjusted magnitude, and Adam 1e-05.
- This is a Fast-80 adaptation of Han et al., not an exact AlexNet/VGG reproduction.
- Selected candidate: `paper_iterative__s50` (50%) — highest paper-inspired sparsity satisfying INT8 validation recall and F1 gates.
- Selection and thresholds use validation only; held-out test is opened after selection.

## Held-out full-INT8 test

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fast-80 pruning baseline | 0.335 | 0.684 | 0.633 | 0.847 | 0.528 | 0.725 | 0.787 |
| Paper-inspired s50 | 0.470 | 0.683 | 0.634 | 0.837 | 0.536 | 0.722 | 0.781 |

## Efficiency interpretation

- Kernel sparsity: 50.0%.
- Raw TFLite: 167,976 B versus 167,976 B controlled re-export.
- Gzip diagnostic: 90,900 B versus 125,325 B.
- Bitmap + INT8 values estimate: 65,590 B for kernels only.
- Theoretical nonzero MACs: 2,935,819.
- Dense executed MACs remain 3,993,536; activation peak remains 38,400 B.
- Sparse acceleration still requires a compatible storage format and custom sparse ESP-NN kernels.

## Physical ESP32-CAM profile

Pending. Run `source scripts/activate_esp_idf.sh && scripts/profile_esp32_model.sh artifacts/pruning/unstructured_magnitude/models/iterative_s50_int8.tflite 80 prune_unstructured_iterative_s50 /dev/cu.usbserial-120`.

## Next technique

Notebook 14 is pattern-based pruning, following the granularity sequence in the pruning plan.
