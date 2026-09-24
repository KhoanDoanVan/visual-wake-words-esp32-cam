# Targeted-36k high-resolution VWW teacher report

- Input: 160×160 RGB with aspect-ratio-preserving letterbox
- Architecture: MobileNetV1 α=0.50, ImageNet initialized
- Training images: 36,000 (subset teacher: 12,000)
- Validation-selected threshold: 0.500
- Test PR-AUC: 0.9301 (subset teacher: 0.9305)
- Test F1: 0.8351 (subset teacher: 0.8464)
- Test specificity: 0.8675 (subset teacher: 0.8685)
- Approved for distillation: **False**

This teacher is not an ESP32 deployment artifact. Compare it with Notebook 01
to measure the value of dataset scale before choosing the source teacher for
student distillation and later INT8 deployment.
