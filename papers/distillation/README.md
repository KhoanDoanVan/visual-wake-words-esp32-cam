# Distillation paper library

Local copies of the primary papers referenced by
[`DISTILLATION_TECHNIQUES_PLAN.md`](../../optimization/distillation/DISTILLATION_TECHNIQUES_PLAN.md).
The numbering follows the recommended reading order: foundations, feature and relation transfer,
quantization-aware distillation, and MCU architecture context.

| # | Local paper | Role in this project | Original source |
|---:|---|---|---|
| 1 | [Distilling the Knowledge in a Neural Network](01_hinton_distilling_knowledge_2015.pdf) | Response/logit KD foundation | [arXiv 1503.02531](https://arxiv.org/abs/1503.02531) |
| 2 | [FitNets: Hints for Thin Deep Nets](02_fitnets_hints_for_thin_deep_nets_2014.pdf) | Intermediate feature hints | [arXiv 1412.6550](https://arxiv.org/abs/1412.6550) |
| 3 | [Paying More Attention to Attention](03_attention_transfer_2016.pdf) | Spatial attention transfer | [arXiv 1612.03928](https://arxiv.org/abs/1612.03928) |
| 4 | [Relational Knowledge Distillation](04_relational_knowledge_distillation_2019.pdf) | Pairwise distance and angle transfer | [arXiv 1904.05068](https://arxiv.org/abs/1904.05068) |
| 5 | [Similarity-Preserving Knowledge Distillation](05_similarity_preserving_kd_2019.pdf) | Pairwise activation similarity | [arXiv 1907.09682](https://arxiv.org/abs/1907.09682) |
| 6 | [Contrastive Representation Distillation](06_contrastive_representation_distillation_2019.pdf) | Contrastive feature transfer | [arXiv 1910.10699](https://arxiv.org/abs/1910.10699) |
| 7 | [Apprentice](07_apprentice_low_precision_kd_2017.pdf) | Distillation for low-precision networks | [arXiv 1711.05852](https://arxiv.org/abs/1711.05852) |
| 8 | [QKD: Quantization-aware Knowledge Distillation](08_qkd_quantization_aware_kd_2019.pdf) | Staged quantization-aware KD | [arXiv 1911.12491](https://arxiv.org/abs/1911.12491) |
| 9 | [MicroNets](09_micronets_tinyml_architectures_2021.pdf) | Closest VWW/TinyML experiment blueprint | [MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/c4d41d9619462c534b7b61d1f772385e-Paper.pdf) |
| 10 | [MCUNet](10_mcunet_tinynas_tinyengine_2020.pdf) | MCU architecture/runtime co-design | [arXiv 2007.10319](https://arxiv.org/abs/2007.10319) |
| 11 | [Visual Wake Words Dataset](11_visual_wake_words_dataset_2019.pdf) | Task and MCU benchmark context | [arXiv 1906.05721](https://arxiv.org/abs/1906.05721) |
| 12 | [Knowledge Distillation: A Survey](12_knowledge_distillation_survey_2020.pdf) | Technique taxonomy and literature map | [arXiv 2006.05525](https://arxiv.org/abs/2006.05525) |

ESP-NN is referenced by the plan as the target kernel library, but it is not included here because
it is a software repository rather than a research paper.

