# NimbusPay Ticket Triage — Technical Report

## 1. Data audit and cleaning

The training split contains 1,445 noisy records. The cleaning pass retains 1,375 rows and records 270 row-level audit actions in `cleaning_log.csv` (`id,action,reason`). Actions include removing placeholders and exact duplicates, rebuilding malformed JSON labels, normalizing legacy priority fields, correcting schema-inconsistent priorities, converting rupee amounts to paise, normalizing dates, and normalizing transaction IDs.

The raw trainable comparison file contains 1,410 parseable rows. Development data (200 labelled records) and test data (400 records) are kept separate from training. A deterministic schema checker matches the development labels 200/200; this is an audit check only, **not model performance**, and is not used at inference.

A normalized customer-text overlap check found 37 development records overlapping with training text; the remaining 163 development records form the no-overlap subset. Every model run moved only slightly when evaluated on the no-overlap subset, and the comparison conclusions were unchanged. In particular, the raw run changed from 74.0% exact match on all 200 to 73.0% on the 163 no-overlap rows. The overlap therefore did not materially inflate the raw result.

## 2. Model and fine-tuning method

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (within the 4B parameter limit).
- **Method:** PEFT LoRA/QLoRA; submitted configuration uses 4-bit QLoRA.
- **Submitted adapter:** rank 16, alpha 16, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- **Trainable parameters:** 18,464,768 (1.18%), below the 50M limit.
- **Training:** 2 epochs, learning rate 2e-4, effective batch size 8, AdamW 8-bit, 3% warmup, maximum sequence length 1,024.
- **Loss masking:** assistant response span only.
- **Inference:** assignment-prescribed system/user chat template, greedy generation, maximum 200 new tokens, whitespace stripping only. No inference-time schema rules, grammar decoding, or field repair.

## 3. Baselines and controlled experiments

The required baselines are the base model with (1) the fixed prompt and (2) schema rules in the prompt. The rules-in-prompt baseline is for comparison only; the final inference prompt does not include those rules.

### Development results

| Run | Exact match (all 200) | Exact match (163 no-overlap) | Mean field accuracy (163) |
|---|---:|---:|---:|
| Cleaned QLoRA, rank 16 (submitted) | 86.5% | 85.9% | 97.5% |
| Raw QLoRA, rank 16 | 74.0% | 73.0% | 95.6% |
| Cleaned 16-bit LoRA, rank 16 | 87.5% | 87.1% | 97.7% |
| Cleaned QLoRA, rank 8 | 63.0% | 62.6% | 93.4% |
| Cleaned QLoRA, rank 16, fraud rows 3x | 86.0% | 84.7% | 97.1% |

Cleaning improves exact match by 12.9 percentage points on the no-overlap subset (85.9% versus 73.0%). Per-field results support the audit findings: priority accuracy is 90.2% for cleaned data versus 81.0% for raw data, while amount_paise is 100% versus 95.1%. Rank 8 has 69.3% priority accuracy, contributing to lower exact match for payment_failed (44.2%) and refund (36.4%).

The 16-bit LoRA result (87.1%) is only 1.2 percentage points above 4-bit QLoRA (85.9%), approximately two records out of 163; this small difference should be treated as within evaluation noise rather than evidence of a decisive precision advantage. Fraud exact match is 75.0% (9/12) in four of the five runs. With only 12 examples, each error changes the rate by about 8.3 percentage points, so this result is indicative, not conclusive.

Peak GPU memory and training minutes should be reported from the executed notebook's measured logs; they are not inferred here. Baseline metrics should likewise be copied from the completed baseline outputs rather than guessed.

## 4. Error analysis and next steps

The final submission should include at least 10 actual development-set model failures in `error_examples.md`, with ticket ID, gold output, model output, incorrect fields, and explanation. Use only real outputs. Priority errors, amount conversion, relative dates, quoted email dates, transaction IDs, legal-threat priority, fraud overrides, Hinglish, and repeat-contact detection are useful error categories to inspect, but should only be described as observed failures when the examples support that conclusion.

With one additional week, prioritize: (1) targeted review of priority-label inconsistencies and amount-in-paise normalization; (2) collecting more representative fraud examples because the current fraud slice is only 12 records; (3) examining rank-8 priority failures before changing model capacity; (4) validating whether the small 16-bit/4-bit gap persists over repeated seeds; and (5) reviewing the 10 documented failures to improve training coverage without using development or test records as training data.

## 5. Submission artifacts

The final repository must include the executed notebook, PEFT adapter files (`adapter_config.json` and `adapter_model.safetensors`), this report, `cleaning_log.csv`, measured `experiment_results.md`, at least 10 real failure examples, and `predictions_test.jsonl` containing exactly 400 records. Do not commit the assignment dataset or base-model weights.
