# Deliverables Evidence Manifest

## Present in repository

- `deliverables/dpo_asses.py` — cleaned training/YAML/preflight/checkpoint/evaluation workflow.
- `SUMMARY.md` — training execution and verification summary.
- `SILENT_FAILURE_AUDIT.md` — explicit silent-failure audit.
- `SOUP_IMPROVEMENTS.md` — proposed verification improvements.
- `README.md` — run configuration and headline findings.

## Evidence already documented

- 450 training samples are reported in the audit summary.
- 3 epochs / 339 steps.
- Tesla T4, approximately 14.6 GB reported memory.
- Checkpoints 200, 300, 339.
- Loss 0.6928 → 0.1326 → 0.0907 → 0.0708.
- LoRA adapter: 128 tensors, 6,815,744 finite/non-zero parameters.
- LoRA runtime: 5/5.
- DPO evaluation: 93/100, 93% preference accuracy, mean margin +0.887668, median +0.765336.
- Base runtime: failed due CUDA/CPU device mismatch; clean base-vs-LoRA generation comparison was skipped.
- Training forensics: 12/13 critical checks passed; trainer-argument preservation was the reported failed check.

## Missing / not independently evidenced in the committed tree

- Raw timestamped training log.
- Raw `nvidia-smi` output/peak telemetry.
- Direct before/after adapter-update measurement.
- Independent per-step gradient/update telemetry.
- Resolved effective sequence length for the run.
- Exact measured peak allocated/reserved VRAM.
- Fresh-process base-vs-LoRA generation comparison.
- Dataset semantic correctness evidence.
- Evaluation train/test leakage or overlap evidence.
- Raw `soup data doctor` output for the exact run.
- Raw `soup ship` output for the exact artifact.
- Completed notebook containing all final verification outputs.

## Submission rule

Do not replace any missing item with an inferred or invented value. A missing artifact should remain explicitly marked missing until the original raw evidence is recovered or the check is rerun.
