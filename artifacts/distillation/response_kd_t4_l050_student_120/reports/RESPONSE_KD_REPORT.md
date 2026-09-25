# Student-120 response-KD report

- Technique: Hinton binary response KD
- MicroNets anchor: T=4, lambda=0.5
- Float PR-AUC delta versus hard control: -0.0008
- Float F1 delta versus hard control: -0.0084
- INT8 probability MAE versus float KD: 0.03556
- Parameters / MACs: 218,801 / 10,487,648
- Static graph identity: **True**
- Accepted before physical profiling: **False**
- Physical device profiling complete: **False**
