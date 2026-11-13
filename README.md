# NimbusPay Support Ticket Triage

ML Intern take-home submission for VIMA3YA.

## Model

- **Base model:** `Qwen2.5-1.5B-Instruct`
- **Fine-tuning:** PEFT 4-bit QLoRA
- **Submitted configuration:** rank 16, alpha 16
- **Trainable parameters:** 18.5M (within the 50M limit)
- **Training:** cleaned data, learning rate 2e-4, 2 epochs, max training length 1,024

## Results

**Submitted model (cleaned QLoRA, rank 16):** 86.5% exact match and 97.7% mean field accuracy on all 200 development records; 85.9% exact match and 97.5% mean field accuracy on the 163 records with no normalized text overlap with training.

Cleaning improves exact match by 12.9 percentage points on the no-overlap subset compared with raw training data (85.9% vs 73.0%). The overlap check changes scores by about one point and does not change the comparison conclusion. See [experiment_results.md](experiment_results.md) and [report.md](report.md).

## Repository contents

- `notebook.ipynb` — final executed notebook with outputs.
- `adapter/` — final PEFT adapter: `adapter_config.json` and `adapter_model.safetensors`.
- `clean_train.py`, `label_rules.py`, `score.py` — audit, cleaning, and evaluation utilities.
- `cleaning_log.csv` — row-level cleaning audit.
- `report.md` — technical report and error analysis.
- `experiment_results.md` — measured development results.
- `error_examples.md` — at least 10 real development-set model failures.
- `predictions_test.jsonl` — one output record for each of the 400 test tickets.

## Data audit

The training split contains 1,445 rows; the cleaning pass retains 1,375 rows and records 270 audit actions. Development and test data are not used for training. The deterministic schema checker is an audit utility only, not an inference-time rule engine.

## Reproduction

Use Google Colab with a Tesla T4 GPU, provide the official assignment archive and helper scripts, and run the notebook from top to bottom. Verify the final adapter, experiment table, error examples, and exactly 400 test predictions before submission.

Do not commit the assignment dataset or base-model weights.
