# Experiment results

Development evaluation results from the completed runs. Full development set: 200 records. No-overlap subset: 163 records (37 normalized customer-text overlaps excluded).

| Run | Exact (200) | Field accuracy (200) | Exact (163) | Field accuracy (163) | Peak GPU (GB)* | Train (min) |
|---|---:|---:|---:|---:|---:|---:|
| cleaned_qlora_r16 (final) | 86.5% | 97.7% | 85.9% | 97.5% | 2.09 | 10.9 |
| raw_qlora_r16 | 74.0% | 95.8% | 73.0% | 95.6% | 3.17 | 11.3 |
| cleaned_lora16_fp16 | 87.5% | 97.8% | 87.1% | 97.7% | 5.10 | 11.5 |
| cleaned_qlora_r8 | 63.0% | 93.6% | 62.6% | 93.4% | 5.51 | 11.0 |
| cleaned_qlora_r16_fraud3x | 86.0% | 97.4% | 84.7% | 97.1% | 4.43 | 12.2 |

*Only the first run's memory measurement is clean; later runs retained GPU allocations from earlier runs. Do not use this column to compare 4-bit and 16-bit memory requirements.

## Interpretation

- Cleaning improves exact match by 12.9 percentage points on the no-overlap subset (85.9% vs 73.0%). Priority accuracy is 90.2% for cleaned data vs 81.0% for raw; amount_paise is 100% vs 95.1%.
- The 37-record overlap check changes scores by about one point and does not change the comparison conclusion. The raw run changes from 74.0% on all 200 to 73.0% on the 163 no-overlap rows; do not claim overlap inflated the raw score.
- 16-bit LoRA is 87.1% vs 85.9% for 4-bit QLoRA on the no-overlap subset, a difference of about two records out of 163 and within noise.
- Rank 8 reaches 62.6% exact on the no-overlap subset; priority accuracy is 69.3%, associated with payment_failed exact match of 44.2% and refund exact match of 36.4%.
- Fraud exact match is 75.0% (9/12) in four runs. With only 12 examples, each miss changes the rate by about 8.3 percentage points. Fraud 3x oversampling does not improve the result.

## Baselines

| Prompt | Valid JSON | Exact match | Mean field accuracy |
|---|---:|---:|---:|
| Base model, fixed system prompt | 1.0% | 0.0% | 0.0% |
| Base model, schema rules in prompt | n/a | 0.0% | 0.6% |
