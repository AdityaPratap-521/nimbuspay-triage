# NimbusPay Support Ticket Triage

ML Intern take-home submission for VIMA3YA.

## Model

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Fine-tuning:** PEFT LoRA / QLoRA
- **Submitted configuration:** 4-bit QLoRA, rank 16
- **Trainable parameters:** approximately 18.46M (confirm against the completed Colab log)
- **Inference:** assignment-prescribed system/user chat format, greedy decoding, and whitespace trimming only.

## Best result

**Pending final executed Colab evaluation.** The best dev exact-match and mean-field-accuracy values will be recorded here only after the completed run. Do not interpret the deterministic schema audit result (200/200) as model accuracy.

## Repository contents

- `notebook.ipynb` — executed end-to-end notebook; replace with the latest corrected Colab notebook before final submission.
- `adapter/` — final PEFT adapter: `adapter_config.json` and `adapter_model.safetensors`.
- `clean_train.py`, `label_rules.py`, `score.py` — data audit/cleaning and evaluation utilities.
- `cleaning_log.csv` — row-level audit trail with `id`, `action`, and `reason`.
- `report.md` — concise technical report with measured results and error analysis.
- `experiment_results.md` — baseline and controlled-experiment measurements.
- `error_examples.md` — at least 10 real development-set model failures.
- `predictions_test.jsonl` — one `{ "id": ..., "output": ... }` record for each of the 400 test tickets.

## Data audit

The training split contains 1,445 rows. The current cleaning pass retains 1,375 rows and records 270 audit actions. The development set is used for evaluation only; it is not used for training. The deterministic schema checker is an audit utility, not an inference-time rule engine.

## Reproduction

1. Use Google Colab with a Tesla T4 GPU.
2. Provide the official assignment archive and helper scripts to the notebook.
3. Run the notebook from top to bottom.
4. Verify the adapter, measured experiment table, error examples, and 400 test predictions before submission.

Do not commit the assignment dataset or base-model weights.
