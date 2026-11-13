# NimbusPay Support Ticket Triage

ML Intern take-home submission for VIMA3YA.

## Model

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Fine-tuning:** PEFT QLoRA
- **Submitted configuration:** 4-bit QLoRA, rank 16
- **Trainable parameters:** 18,464,768 (1.18%)
- **Best measured submitted-run result:** 85.9% exact match and 97.5% mean field accuracy on the 163-record no-overlap development subset (86.5% exact match on all 200).

## Results summary

Cleaning improved exact match by 12.9 percentage points on the no-overlap subset compared with raw training data (85.9% vs 73.0%). The 37-record overlap check changed all runs only slightly and did not change the comparison conclusion. See [experiment_results.md](experiment_results.md) and [report.md](report.md).

## Repository contents

- `notebook.ipynb` — executed notebook; ensure this is the final run with outputs.
- `adapter/` — final PEFT adapter: `adapter_config.json` and `adapter_model.safetensors`.
- `clean_train.py`, `label_rules.py`, `score.py` — audit, cleaning, and evaluation utilities.
- `cleaning_log.csv` — row-level cleaning audit.
- `report.md` — technical report and analysis.
- `experiment_results.md` — measured development results and interpretation.
- `error_examples.md` — at least 10 real development-set model failures.
- `predictions_test.jsonl` — one output record for each of the 400 test tickets.

## Data audit

The training split contains 1,445 rows; the current cleaning pass retains 1,375 rows and records 270 audit actions. Development and test data are not used for training. The deterministic schema checker is an audit utility only, not an inference-time rule engine.

## Reproduction

Use Google Colab with a Tesla T4 GPU, provide the official assignment archive and helper scripts, and run the notebook top to bottom. Verify the final adapter, measured experiment table, error examples, and exactly 400 test predictions before submission.

Do not commit the assignment dataset or base-model weights.
