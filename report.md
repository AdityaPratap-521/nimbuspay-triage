# NimbusPay triage: audit, experiments, error analysis

**Submitted model:** Qwen2.5-1.5B-Instruct, 4-bit QLoRA, rank 16 (18.5M trainable parameters), cleaned data, learning rate 2e-4, 2 epochs, training length 1,024. This is the pre-registered configuration; I retained it although the 16-bit run scored slightly higher. Inference uses the fixed system prompt, greedy decoding and `.strip()` only; no schema rules or field repair.

**Result:** 86.5% exact match and 97.7% mean field accuracy on all 200 dev rows; 85.9% exact match and 97.5% field accuracy on the 163 rows with no normalized text overlap with training.

## 1. Data audit (Part A)

The training split has 1,445 rows; 1,375 are retained and 270 actions are recorded in `cleaning_log.csv`. The audit found placeholders (10 dropped), exact duplicates (60 dropped), truncated assistant JSON (25 rebuilt), legacy `prio` labels (70 normalized), priority contradictions (60 corrected), rupee amounts/fraction loss (25 corrected), date formats (30 normalized), and transaction IDs needing normalization (20). These are audit counts by issue and may overlap. Example errors included tr-00013 (KYC incorrectly labelled P1), tr-00144 (INR 167 represented as 167 instead of 16,700 paise), and `np 81633 47892` normalized to `NP` plus 10 digits.

About 200 category paraphrases missed by the deterministic checker were sampled; their original categories were retained where labels were correct. The checker matched 200/200 dev labels, but this is only a checker audit—not model performance—and was not used at inference or to edit training rows.

**Overlap:** 37 of 200 dev customer texts matched training text after normalization; none of 400 test texts did. Cleaning used training data and SCHEMA.md only; dev was not used to clean training. Results are shown on all 200 dev rows and the 163 non-overlapping rows. The overlap moved scores by about one point and did not change conclusions.

**Length:** Median ticket length is 196 characters, p95 325; 40 quoted-thread tickets are 4,000–8,800 characters. Training truncates at 1,024 tokens. Inference uses a 4,096-token context because some dev inputs were about 2,100 tokens and generation failed at 1,024; test inputs are not shortened.

## 2. Fine-tuning setup (Part C)

Qwen2.5-1.5B-Instruct was fine-tuned with Unsloth and PEFT QLoRA. LoRA targets q/k/v/o/gate/up/down projections, alpha equals rank, effective batch size is 8 (2 per device × 4 accumulation), optimizer AdamW 8-bit, warmup 3%, fp16 on the T4. Loss is applied only to the assistant span; the mask guard rejects empty masks or masks over 65% of tokens (observed 39.6%). The fixed system prompt and model chat template are used.

## 3. Baselines (Part B)

| Base-model prompt | Valid JSON | Exact match | Mean field accuracy |
|---|---:|---:|---:|
| Fixed system prompt | 1.0% | 0.0% | 0.0% |
| Schema rules in prompt (reference only) | n/a | 0.0% | 0.6% |

The untuned 1.5B model did not reliably produce the required JSON format. The schema-rules prompt is a baseline only; those rules are not included in final inference.

## 4. Controlled experiments (Part D)

| Run | Exact (200) | Field (200) | Exact (163) | Field (163) | Peak GB* | Train min |
|---|---:|---:|---:|---:|---:|---:|
| Cleaned QLoRA r16 (submitted) | 86.5% | 97.7% | 85.9% | 97.5% | 2.09 | 10.9 |
| Raw QLoRA r16 | 74.0% | 95.8% | 73.0% | 95.6% | 3.17 | 11.3 |
| Cleaned 16-bit LoRA r16 | 87.5% | 97.8% | 87.1% | 97.7% | 5.10 | 11.5 |
| Cleaned QLoRA r8 | 63.0% | 93.6% | 62.6% | 93.4% | 5.51 | 11.0 |
| Cleaned QLoRA r16, fraud ×3 | 86.0% | 97.4% | 84.7% | 97.1% | 4.43 | 12.2 |

*Only the first run's GPU memory measurement is clean; later runs retained allocations from previous runs, so memory figures cannot fairly compare 4-bit and 16-bit. Results use one seed; small differences are uncertain.

Cleaning improves exact match by 12.9 points on the 163 non-overlap rows. Priority accuracy is 90.2% cleaned vs 81.0% raw; amount_paise is 100% vs 95.1%. The 16-bit vs 4-bit gap is 1.2 points (about two tickets), within noise, so the pre-registered QLoRA model remains submitted. Rank 8 has 69.3% priority accuracy; the reason for its decline is a hypothesis, not isolated by this experiment. Fraud ×3 gives no improvement.

## 5. Error analysis (Part E)

Weakest fields are priority (90.2%) and category (97.5%) on the no-overlap subset. Weakest categories by exact match are fraud (75.0%, 9/12), account_access (73.3%), payment_failed (81.4%) and refund (84.1%). Fraud has a small sample; each miss changes its rate by about 8.3 points.

The following are the first 10 final-model dev failures in dev order, not a random sample. Nine involve priority; five needs_human errors follow from an incorrect priority because needs_human is true for P1 or repeat contact.

| ID | Gold → prediction (wrong fields) | Ticket evidence and explanation |
|---|---|---|
| dv-00034 | priority P2→P1; needs_human F→T | Declined wallet payment, Rs 36,737. Model over-called P1 below the Rs 50,000 threshold. |
| dv-00040 | priority P2→P1; needs_human F→T | Missing refund, Rs 33,986, “URGENT!!”. Urgency wording may have pushed P1, though it is not a schema rule. |
| dv-00056 | priority P2→P1; needs_human F→T | Double charge, Rs 44,206.75; another P1 over-call below Rs 50,000. |
| dv-00081 | priority P2→P1; needs_human F→T | Card payment failed, Rs 24,444; amount is above the Rs 5,000 P2 threshold only. |
| dv-00037 | priority P1→P3; needs_human T→F | Double charges, Rs 284,669; amount extracted correctly but P1 threshold missed. |
| dv-00044 | priority P2→P3 | Refund Rs 8,377 and “second complaint”; missed the Rs 5,000 refund threshold. |
| dv-00054 | priority P2→P3 | Hinglish payment-stuck ticket, Rs 6,893; amount extracted correctly but P2 threshold missed. |
| dv-00083 | priority P2→P3 | App keeps logging out; category was correct but account_access→P2 rule was not applied. Typo and long signature may have contributed. |
| dv-00046 | category fraud→refund; priority P1→P3 | Disputed statement payment, earlier ticket, repeated contact. Fraud relies on paraphrase rather than explicit “I did not make this”; priority error follows category error. |
| dv-00087 | channel card→null | “My new card” is explicit; model left channel null, applying the explicit-channel requirement too strictly. |

The pattern suggests the model extracts amounts well but applies the Rs 5,000 and Rs 50,000 thresholds inconsistently. This is a hypothesis; no experiment isolated the cause.

## 6. With one more week

I would create additional training-only examples around the Rs 5,000 and Rs 50,000 thresholds, recomputing priority and needs_human with the schema checker. I would repeat rank, precision and fraud experiments with three seeds, measure GPU memory in fresh processes, and evaluate long quoted-thread tickets separately. I would also add schema-faithful fraud paraphrases and test a model up to 3B parameters. Dev and test would remain evaluation-only.

## 7. AI tools used

Cursor assistant (Grok) helped create early versions of `label_rules.py`, `clean_train.py`, the notebook and report. Claude helped debug the context-length crash and notebook, review the overlap analysis and draft report text. ChatGPT helped troubleshoot, review results and prepare the final report and submission. I ran the experiments, reviewed the outputs and am responsible for the reported numbers. No AI produced labels, test predictions or model weights.
