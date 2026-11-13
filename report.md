# NimbusPay Ticket Triage — Technical Report

**Submission status:** Training and evaluation are being completed in Google Colab. Model metrics and error examples below must be populated from the executed notebook before final submission.

## 1. Data audit and cleaning

The training split contains 1,445 noisy records; the cleaned training file contains 1,375. The cleaning audit records 270 row-level actions in `cleaning_log.csv` with the required columns `id,action,reason`. Actions include removing placeholder tickets and exact duplicates, rebuilding malformed JSON labels, normalizing legacy priority fields, correcting schema-inconsistent priorities, converting rupee amounts to paise, normalizing dates, and normalizing transaction IDs.

The raw trainable comparison file contains 1,410 parseable rows. Development data (200 labelled records) and test data (400 records) are kept separate from training. A deterministic checker in `label_rules.py` matches the development labels 200/200; this is a schema-audit check only and **is not model performance** and is not used at inference.

A normalized customer-text comparison identified 37 development records also present in training; none of the 400 test texts overlapped. These development rows are disclosed, not used to modify training. Report both full-dev results (200 rows) and the no-overlap subset (163 rows) where the executed notebook provides them.

## 2. Model and fine-tuning method

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (within the 4B parameter limit).
- **Method:** PEFT LoRA/QLoRA; submitted configuration uses 4-bit QLoRA.
- **Adapter:** rank 16, alpha 16, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- **Trainable parameters:** 18,464,768 (1.18%) as reported by the rank-16 training log; below the 50M limit.
- **Training:** 2 epochs, learning rate 2e-4, effective batch size 8, AdamW 8-bit, 3% warmup, maximum sequence length 1,024.
- **Loss masking:** assistant response span only.
- **Inference:** assignment-prescribed system/user chat template, greedy generation, maximum 200 new tokens, whitespace stripping only. No inference-time schema rules, grammar decoding, or field repair.

The notebook records the actual GPU memory and training time for each completed run. The run log and exported adapter must be retained with the submission.

## 3. Baselines and controlled experiments

The required baselines are the base model with (1) the fixed prompt and (2) schema rules in the prompt. The rules-in-prompt baseline is a comparison only; the final inference prompt does not include those rules.

Four controlled comparisons are run, changing one factor at a time: raw versus cleaned training data; 16-bit LoRA versus 4-bit QLoRA; rank 8 versus rank 16; and a fraud-row 3x oversampling experiment. Record dev exact match, mean field accuracy, peak GPU memory, and training minutes for every completed run. Failed/OOM runs must be reported honestly.

| Run | Dev exact match | Mean field accuracy | Peak GPU memory | Minutes |
|---|---:|---:|---:|---:|
| Base, fixed prompt | pending | pending | pending | pending |
| Base, schema rules in prompt | pending | pending | pending | pending |
| Cleaned 4-bit QLoRA, rank 16 (submitted) | pending | pending | pending | pending |
| Raw vs cleaned | pending | pending | pending | pending |
| 16-bit LoRA vs 4-bit QLoRA | pending | pending | pending | pending |
| Rank 8 vs rank 16 | pending | pending | pending | pending |
| Fraud rows repeated 3x | pending | pending | pending | pending |

These values are intentionally left pending until copied from the executed Colab results. The deterministic checker score above must not be substituted for model metrics.

## 4. Error analysis and next steps

The final report must include at least 10 actual development-set model failures, with the ticket ID, gold output, model output, incorrect fields, and a short explanation. Likely high-difficulty cases to inspect include relative dates and quoted email dates, rupee-to-paise conversion, transaction IDs, legal-threat priority, fraud overrides, Hinglish, and repeat-contact detection. These are analysis categories, not claims about observed model failures; populate them only where supported by the model outputs.

With one additional week, prioritize reviewing train paraphrases that are difficult to classify, adding schema-faithful examples from training data only, testing a second model size within the 4B limit, and improving weak fields based on the measured error breakdown. Do not train on development or test data.

## 5. Submission artifacts

The completed repository must include the executed notebook, PEFT adapter files (`adapter_config.json` and `adapter_model.safetensors`), this report, `cleaning_log.csv`, measured `experiment_results.md`, at least 10 real failure examples, and `predictions_test.jsonl` containing exactly 400 records. The dataset and base-model weights must not be committed.
