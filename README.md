# SOUP — Russian DPO Training Audit

Engineering audit of a Russian-language DPO/LoRA training run performed
with SOUP on a Tesla T4.

## Run

- Base model: `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24`
- Task: DPO
- Backend: Transformers
- Quantization: 4-bit
- LoRA rank: 16
- LoRA alpha: 32
- LoRA targets: `q_proj`, `v_proj`
- Dataset: 450 training samples
- Training: 3 epochs / 339 steps
- Hardware: Tesla T4 / 14.6 GB reported memory

## Main finding

The run contains substantial evidence that training actually executed:
timestamped training logs, progressive checkpoints, trainer state,
changing loss and learning-rate histories, and a non-zero finite LoRA
adapter.

However, the audit also exposed an important verification gap.

Metrics such as loss, DPO reward accuracy, and checkpoint existence are
not sufficient by themselves to establish that the resulting model can
be correctly loaded and executed.

During independent verification, the existing base model had an
incompatible CPU/disk offload state and could not successfully generate.
The LoRA model could generate successfully.

This is therefore an audit of both a successful training run and a
verification weakness in the surrounding workflow.

## Result

- Checkpoint: PASS
- LoRA configuration: PASS
- LoRA parameters: PASS
- Tokenizer: PASS
- LoRA runtime: 5/5 PASS
- DPO evaluation: 93/100 preference accuracy
- Base-vs-LoRA runtime comparison: skipped because the existing base
  model's offload state was not executable
- Training forensics: 12/13 critical evidence checks passed

The single failed forensic check was preservation of trainer arguments
in the inspected trainer state.

## Engineering lesson

SOUP should move beyond metric-only validation and provide event-based,
model-aware verification:

1. Validate whether the selected model can fit the current hardware.
2. Record the exact resolved training configuration.
3. Verify that intended trainable parameters actually receive updates.
4. Verify checkpoints independently.
5. Test loading the produced artifact in a fresh process.
6. Test actual inference, not merely artifact existence.
7. Detect CPU/disk/meta-device offload failures.
8. Record independent hardware telemetry.
9. Separate training evidence from model-quality evidence.
10. Produce actionable diagnostics when a run cannot be safely verified.

See `SUMMARY.md` and `SOUP_IMPROVEMENTS.md`.
