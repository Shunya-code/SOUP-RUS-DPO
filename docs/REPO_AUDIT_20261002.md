# Repository Audit — 2026-10-02

## Checks performed

- JSON and YAML syntax validation: passed.
- JSONL validation: passed for train (500), test (261), validation (50), assessment (261), and diagnostic (50).
- Train/test exact-record overlap: 0.
- Train/test prompt overlap: 0.
- Validation-50 source: intentionally sampled from train with seed 42.
- Assessment-261 source: exactly the held-out test split.
- Legacy duplicate verification files: byte-for-byte identical to canonical copies.
- Legacy duplicate result/evidence directories: consolidated into the canonical `Evaluations/<model>/` tree.
- Parameter-update verification: **PASS** on retained checkpoint-200 → checkpoint-339; raw output is committed. Checkpoint directories remain intentionally excluded from Git.

## Important provenance note

Some JSON manifests retain absolute Google Drive paths from the original Colab run. Those paths are historical provenance, not repository-relative paths. They are intentionally not rewritten inside raw evidence artifacts.

## Parameter-update verification result

The retained Drive checkpoints were compared with `deliverables/verify_parameter_updates.py`. The result is **PASS**: 128/128 LoRA tensors changed and 99.999061% of adapter elements changed between checkpoint-200 and checkpoint-339. Raw output is committed at `Evaluations/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/results/parameter_update_verification.txt`.
