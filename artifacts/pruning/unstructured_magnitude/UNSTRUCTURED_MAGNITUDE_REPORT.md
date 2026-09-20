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

- Exact selected-model profile: `artifacts/device_profiles/prune_unstructured_iterative_s50`.
- Batch 1, 20 measured invokes after 2 warm-ups.

| Metric | Fast-80 | Iterative-s50 | Change |
|---|---:|---:|---:|
| Full Invoke mean | 455.546 ms | 416.805 ms | -8.5% |
| TFLM arena used | 69,068 B | 69,068 B | +0 B |
| Live activation peak | 38,400 B | 38,400 B | +0 B |
| Dense MACs executed | 3,993,536 | 3,993,536 | unchanged |
| Model bytes | 167,976 B | 167,976 B | unchanged |

- Candidate live pipeline: 1.68 fps, 59.8 ms median preprocessing, and 597.0 ms median frame period.
- The observed Invoke difference comes from separate physical capture sessions. Because the FlatBuffer shapes, dense MAC count, and arena are unchanged, it must not be attributed to unstructured zeros without repeated alternating trials or a sparse ESP-NN kernel.
- White flash was compiled in but disabled with `kFlashLedEnabled = false`; boot confirmed GPIO4 was forced low.

## Next technique

Notebook 14 is pattern-based pruning, following the granularity sequence in the pruning plan.
