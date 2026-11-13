# Experiment results

This file is a reporting template until values are populated from completed notebook runs. Do not estimate or invent metrics.

## Baselines

| Baseline | Dev exact match (200) | Dev mean field accuracy | Peak GPU memory (GB) | Minutes |
|---|---:|---:|---:|---:|
| Base model, fixed prompt | pending | pending | pending | pending |
| Base model, schema rules in prompt | pending | pending | pending | pending |

## Controlled experiments

| Run / single change | Dev exact match (200) | Dev mean field accuracy | Peak GPU memory (GB) | Training minutes | Notes |
|---|---:|---:|---:|---:|---|
| Cleaned data, 4-bit QLoRA, rank 16 (submitted) | pending | pending | pending | pending | Main adapter |
| Raw trainable data vs cleaned data | pending | pending | pending | pending | Data-quality comparison |
| 16-bit LoRA vs 4-bit QLoRA | pending | pending | pending | pending | Record OOM honestly if it occurs |
| Rank 8 vs rank 16 | pending | pending | pending | pending | Capacity comparison |
| Fraud rows repeated to 3x | pending | pending | pending | pending | Class-imbalance experiment |

## Interpretation

Complete this section only after the corresponding runs finish. Explain measured trade-offs. If a run fails, record the actual failure rather than inventing a score.
