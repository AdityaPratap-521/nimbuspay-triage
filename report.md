# NimbusPay triage: audit, experiments, error analysis

**Submitted model:** Qwen2.5-1.5B-Instruct, 4-bit QLoRA, rank 16 (18.5M trainable params), cleaned data, lr 2e-4, 2 epochs, max training length 1024. This is the pre-registered configuration; I kept it as the submitted model although the 16-bit run scored 1.2 percentage points higher on the 163-row no-overlap subset. Rules are learned from data, never put in the inference prompt; the only post-processing is `.strip()`.

**Result (dev):** 86.5% exact match, 97.7% mean field accuracy on all 200 rows; 85.9% / 97.5% on the 163 rows with no text overlap with train.

## 1. Audit (Part A)

`train.jsonl` has 1,445 rows; 1,375 kept, 270 log lines (`cleaning_log.csv`). The checker `label_rules.py` implements SCHEMA.md and matches 200/200 dev labels (used to check the checker only, never at inference and never to edit training rows from dev).

| Problem | Evidence | Action | Rows |
|---|---|---|---:|
| Placeholder text (test, asdf, empty) | Body check | Drop | 10 |
| Exact duplicates | Same text and label | Drop later ID | 60 |
| Truncated assistant JSON | Fails to parse | Rebuild from ticket; category kept from cut string | 25 |
| Legacy `prio` low/medium/high | Missing `priority` key | Recompute P1/P2/P3 | 70 |
| Priority contradicts rules | e.g. tr-00013 KYC, no amount, labelled P1 | Replace | 60 |
| Amount in rupees or fraction lost | e.g. tr-00144 `167` for "INR 167" (want 16700) | Replace | 25 |
| Date not YYYY-MM-DD | e.g. 02/09/2026 | Normalise | 30 |
| Transaction ID not normalised | e.g. `np 81633 47892` | Normalise to `NP` + 10 digits | 20 |

Categories on rows that already parsed were kept: about 200 paraphrases the checker missed (e.g. "unfamiliar payment", "won't go through") were sampled and the original label was right.

**Train/dev text overlap (disclosed, not cleaned).** 37 of 200 dev customer texts also occur in train (normalised match); 0 of 400 test texts do. Cleaning is limited to `train.jsonl` and SCHEMA.md, so dev was not used to drop or fix training rows. Scores are reported on all 200 rows and on the 163 without overlap. The overlap moved scores by about 1 point and changed no conclusion; it did not materially inflate the raw run.

**Length.** Median ticket 196 characters, p95 325; 40 tickets are 4,000–8,800 characters of quoted threads (customer text on top). Training truncates at 1,024 tokens. Inference uses a 4,096-token context because some dev tickets are about 2,100 tokens and generation failed at 1,024; test inputs are not shortened.

## 2. Setup (Part C)

Unsloth for 4-bit loading and speed; adapter saved in PEFT format. LoRA targets q, k, v, o, gate, up and down projections; alpha = rank, batch 2 × 4 accumulation, AdamW 8-bit, warmup 3%, fp16 (T4 has no bf16). Loss is applied only to the assistant span (`<|im_start|>assistant`); the run aborts if the mask is empty or covers over 65% of tokens (measured 39.6%). Qwen has a system role, so the fixed system prompt is used unchanged.

## 3. Baselines (Part B), dev, base model 4-bit

| Prompt | Valid JSON | Exact | Mean field accuracy |
|---|---:|---:|---:|
| Fixed system prompt | 1.0% | 0.0% | 0.0% |
| Rules written into prompt (reference only) | n/a | 0.0% | 0.6% |

The untuned 1.5B base model did not reliably follow the required output format, so almost every field was scored wrong. The rules-in-prompt run is a reference baseline only; the rules are not used for final inference.

## 4. Experiments (Part D)

One variable changed per comparison. Results use one seed; differences under about 2.5 percentage points on 200 rows should be treated cautiously.

| Run | Change | Exact (200) | Field (200) | Exact (163) | Field (163) | Peak GB* | Train min |
|---|---|---:|---:|---:|---:|---:|---:|
| cleaned_qlora_r16 (final) | Cleaned data, 4-bit QLoRA, rank 16 | 86.5% | 97.7% | 85.9% | 97.5% | 2.09 | 10.9 |
| raw_qlora_r16 | Raw data instead of cleaned | 74.0% | 95.8% | 73.0% | 95.6% | 3.17 | 11.3 |
| cleaned_lora16_fp16 | 16-bit LoRA instead of 4-bit | 87.5% | 97.8% | 87.1% | 97.7% | 5.10 | 11.5 |
| cleaned_qlora_r8 | Rank 8 instead of 16 | 63.0% | 93.6% | 62.6% | 93.4% | 5.51 | 11.0 |
| cleaned_qlora_r16_fraud3x | Fraud rows repeated 3x | 86.0% | 97.4% | 84.7% | 97.1% | 4.43 | 12.2 |

*Only the first run's memory measurement is clean; later runs retained GPU allocations from earlier runs, so the memory column must not be used to compare 4-bit and 16-bit memory requirements.

- **Cleaning helps:** +12.9 points exact match on the 163 no-overlap rows. The gain is mainly `priority` (90.2% vs 81.0%) and `amount_paise` (100% vs 95.1%), fields affected by audit corrections.
- **16-bit vs 4-bit:** 87.1% vs 85.9% exact match on 163 rows (about two tickets), within noise. The pre-registered 4-bit QLoRA configuration remains the submitted model.
- **Rank 8:** much worse (62.6% exact match), with `priority` accuracy at 69.3%. The reason for the rank-8 drop is a hypothesis, not isolated by this experiment.
- **Fraud 3x:** no gain, so this oversampling change did not improve the measured fraud result.

## 5. Error analysis (Part E)

The supplied field breakdown identifies `priority` as a weak field (90.2% on the no-overlap subset for the cleaned run). `needs_human` also deserves targeted review because it depends on detecting P1 conditions and repeat-contact language; no separate accuracy figure for this field was supplied here, so none is claimed. By category, fraud is 75.0% (9/12; same exact-match rate in four of five runs), account_access is 73.3%, and payment_failed is 81.4%. The fraud slice is small: each miss changes its rate by about 8.3 percentage points.

**Ten specific failures** (from `error_examples.md`):
[Awaiting the actual ten examples. No ticket-level failures or explanations have been fabricated.]

## 6. With one more week

I would first review the actual `priority` and `needs_human` failures and add more varied training-only examples for legal threats, high-value disputed transactions, fraud overrides, and repeat-contact wording. I would then inspect account_access and payment_failed errors, improve coverage of their difficult phrasing, and re-run the same evaluation with multiple seeds to determine whether the small 16-bit/4-bit difference is reproducible. I would also add a larger, representative fraud evaluation set before drawing conclusions from the current 12 examples. All changes would use training data only; dev would remain an evaluation set and test would remain untouched.

## 7. AI tools used

Cursor assistant (Grok) was used for early versions of `label_rules.py`, `clean_train.py`, the notebook and report. Claude was used for debugging the context-length issue and notebook fixes, and for reviewing the dev-overlap analysis. ChatGPT was used for troubleshooting, code/notebook and report review, and submission preparation. I ran and reviewed the experiments and am responsible for the reported numbers and submitted work. No AI assistant or external model/API produced training labels, test predictions, or model weights.
