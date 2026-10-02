# Experiment Card

> Russian-language DPO + LoRA experiment using SOUP on a Tesla T4.

| Item | Value |
|---|---|
| Base model | `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24` |
| Method | DPO + LoRA |
| Quantization | 4-bit |
| Supplied training data | 500 examples |
| Soup training split | 450 train + 50 validation |
| Independent assessment | 261 held-out test examples |
| Epochs | 3 |
| Steps | 339 |
| Learning rate | `5e-6` |
| LoRA | `r=16`, `alpha=32`, dropout `0.05` |
| Targets | `q_proj`, `v_proj` |
| GPU | Tesla T4, ~14.6 GiB |
| Final adapter | 6,815,744 parameters / 128 tensors |

## Evaluation snapshot

| Evaluation | Base | LoRA | Change |
|---|---:|---:|---:|
| 50-example validation | 86.0% | 90.0% | +4.0 pp |
| 261-example independent assessment | 87.74% | 87.36% | -0.38 pp |
| 261-example prompted-base control | 87.74% | 89.27% | +1.53 pp |

The 93/100 DPO preference score is a **training-set verification measurement**, not a held-out generalization result.

## Verification status

- Training execution evidence: present
- Checkpoint progression: present
- Adapter integrity: present
- LoRA runtime generation: 5/5
- Independent assessment: present
- Base-vs-LoRA generation comparison: invalid in the original runtime because the base model had an incompatible CPU/disk offload state
- Parameter-update verification across checkpoint-200 → checkpoint-339: **PASS** — 128/128 tensors changed; 99.999061% of adapter elements changed
- Independent raw GPU telemetry: preserved for the forensic run where available

## Interpretation

The experiment demonstrates that SOUP executed a DPO/LoRA training run and produced a valid, non-empty adapter. The independent 261-example assessment does **not** show an aggregate accuracy improvement. The useful engineering result is therefore the complete evidence trail and the identification of verification gaps, rather than a claim of generalization improvement.
