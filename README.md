# NimbusPay Support Ticket Triage

ML Intern take-home submission for VIMA3YA.

## Model

- **Base model:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** PEFT 4-bit QLoRA
- **Submitted configuration:** rank 16, alpha 16
- **Trainable parameters:** 18.5M (within the 50M limit)
- **Training:** cleaned data, learning rate 2e-4, 2 epochs, max training length 1,024

## Results

**Submitted model (cleaned QLoRA, rank 16):** 86.5% exact match and 97.7% mean field accuracy on all 200 development records; 85.9% exact match and 97.5% mean field accuracy on the 163 records with no normalized text overlap with training.

Cleaning improves exact match by 12.9 percentage points on the no-overlap subset compared with raw training data (85.9% vs 73.0%). The overlap check changes scores by about one point and does not change the comparison conclusion. See [report.md](report.md) for the audit, baselines, experiments and error analysis.

## Submission files

- `notebook.ipynb` — executed notebook with outputs.
- `adapter/` — final PEFT adapter containing `adapter_config.json` and `adapter_model.safetensors`.
- `report.md` — technical report (maximum three pages).
- `cleaning_log.csv` — row-level audit trail.
- `predictions_test.jsonl` — one prediction for each of the 400 test records.

## Existing project support files

The repository also retains the existing `clean_train.py`, `cleaning_summary.json`, `label_rules.py`, and `score.py` used by the notebook. The cleaning summary records 1,445 original training rows, 1,375 cleaned rows, and 270 audit-log entries.

Do not commit the assignment dataset or base-model weights.
