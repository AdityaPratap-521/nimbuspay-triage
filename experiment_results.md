# Experiment results

Development evaluation results supplied from the completed runs. Full development set: 200 records. No-overlap subset: 163 records (37 text-overlap records excluded).

| Run | Exact match (all 200) | Exact match (163 no-overlap) | Mean field accuracy (163) | Peak GPU memory (GB) | Training minutes |
|---|---:|---:|---:|---:|---:|
| cleaned_qlora_r16 (submitted) | 86.5% | 85.9% | 97.5% | see executed notebook | see executed notebook |
| raw_qlora_r16 | 74.0% | 73.0% | 95.6% | see executed notebook | see executed notebook |
| cleaned_lora16_fp16 | 87.5% | 87.1% | 97.7% | see executed notebook | see executed notebook |
| cleaned_qlora_r8 | 63.0% | 62.6% | 93.4% | see executed notebook | see executed notebook |
| cleaned_qlora_r16_fraud3x | 86.0% | 84.7% | 97.1% | see executed notebook | see executed notebook |

## Interpretation

- Cleaning gains 12.9 percentage points of exact match on the no-overlap subset (85.9% vs 73.0%).
- The overlap check does not change the comparison: the raw run shifts from 74.0% on all 200 to 73.0% on the 163 no-overlap records, similar to the small shifts in other runs. Do not characterize overlap as inflating the raw score.
- Priority accuracy is 90.2% for cleaned data vs 81.0% for raw data. Amount_paise accuracy is 100% vs 95.1%.
- Rank 8 priority accuracy is 69.3%, associated with payment_failed exact match of 44.2% and refund exact match of 36.4%.
- Fraud exact match is 75.0% (9/12) in four runs. The sample is small; each miss moves the rate by about 8.3 percentage points.
- The 16-bit vs 4-bit no-overlap exact-match gap is 1.2 percentage points (87.1% vs 85.9%), about two examples out of 163 and not decisive by itself.

Peak GPU memory and training minutes must be copied from the executed notebook logs. Do not invent them. Baseline scores should also be added from their actual output files.
