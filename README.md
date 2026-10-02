# 🍲 SOUP-RUS-DPO

### Russian DPO + LoRA training experiment, evaluation, and verification audit

[![Experiment](https://img.shields.io/badge/experiment-DPO%20%2B%20LoRA-blue)](./docs/EXPERIMENT_CARD.md)
[![Language](https://img.shields.io/badge/language-Russian-red)](./Data/train.jsonl)
[![Hardware](https://img.shields.io/badge/GPU-Tesla%20T4-green)](./docs/REPRODUCTION.md)
[![Status](https://img.shields.io/badge/status-audited-orange)](./SILENT_FAILURE_AUDIT.md)

A reproducible engineering record of a Russian-language **Direct Preference Optimization (DPO)** run performed with **SOUP**, using a 4-bit 8B base model and a LoRA adapter.

The repository is deliberately organized around one question:

> **Did training actually happen, what changed, and how much of that change can we independently verify?**

---

## ⚡ At a glance

| | Result |
|---|---:|
| Base model | `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24` |
| Training method | DPO + LoRA |
| Quantization | 4-bit |
| Supplied train set | 500 examples |
| SOUP split | 450 train / 50 validation |
| Independent assessment | 261 examples |
| Training | 3 epochs / 339 steps |
| GPU | Tesla T4 / ~14.6 GiB |
| Adapter | 6,815,744 parameters / 128 tensors |
| LoRA runtime | 5/5 successful generations |

### The important numbers

| Evaluation | Base | LoRA | Change |
|---|---:|---:|---:|
| 50-example validation | 86.0% | **90.0%** | **+4.0 pp** |
| 261-example independent assessment | 87.74% | **87.36%** | **-0.38 pp** |
| Prompted-base control, 261 | 87.74% | **89.27%** | **+1.53 pp** |

The 261-example assessment is the strongest independent comparison in this repository. It does **not** show an aggregate accuracy improvement from the LoRA adapter.

The separate **93/100 DPO preference score** is calculated on the training dataset and is therefore treated as training-set verification, not held-out generalization.

---

## 🔬 What happened

The training run produced a substantial evidence trail:

- timestamped training log
- checkpoints at 200, 300, and 339 steps
- trainer state and learning-rate history
- changing training loss
- LoRA configuration
- 6.8M finite, non-zero adapter parameters
- successful LoRA runtime generation
- independent 50-example validation
- independent 261-example assessment
- base-vs-prompted control
- forensic and failure-analysis artifacts

The audit also found a real verification problem: the original base model had an incompatible CPU/disk offload state and could not be used for a clean base-vs-LoRA generation comparison in that runtime.

That failure is recorded as an **environment/verification failure**, not as evidence that the trained adapter is bad.

---

## 📊 Evaluation evidence

The assessment contains per-example margins, transition analysis, and visualizations.

**261-example transitions:**

- Base correct → LoRA correct: **218**
- Base wrong → LoRA wrong: **22**
- Base correct → LoRA wrong: **11**
- Base wrong → LoRA correct: **10**

So the adapter recovered 10 examples but lost 11 previously correct examples on the independent assessment.

Visual analysis is available under:

`Evaluations/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/results/analysis/`

---

## 🧪 Verification status

### Confirmed

- [x] Training process executed to completion
- [x] Checkpoint progression exists
- [x] LoRA configuration is present
- [x] Adapter tensors are finite and non-zero
- [x] LoRA runtime generation: 5/5
- [x] Train/test records are structurally valid
- [x] Train/test have no exact-record overlap
- [x] Train/test have no prompt overlap
- [x] Independent 261-example assessment preserved

### Still open

- [ ] Compare actual LoRA tensor deltas between checkpoint-200 and checkpoint-339
- [ ] Fresh-process base-vs-LoRA generation comparison
- [ ] Independent proof that gradients/optimizer updates occurred at every intended step
- [ ] Full semantic dataset-quality audit
- [ ] Stronger leakage analysis beyond exact/prompt overlap

Parameter-update verification is now complete: checkpoint-200 → checkpoint-339 changed 128/128 LoRA tensors, with 99.999061% of adapter elements changing. The raw verification output is preserved at `Evaluations/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/results/parameter_update_verification.txt`. The verification script remains available at `deliverables/verify_parameter_updates.py`.

It must be run against the checkpoints retained outside Git.

---

## 🗂️ Repository map

```text
SOUP-RUS-DPO/
│
├── Data/                         # Dataset inputs
│   ├── train.jsonl
│   ├── test.jsonl
│   └── rusian_data(dpo).csv
│
├── Models/                       # Model + adapter artifacts
│   └── Vikhr-Llama3.1-8B-Instruct-R-21-09-24/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer.*
│       ├── chat_template.jinja
│       └── soup.yaml
│
├── Evaluations/                  # Canonical experiment evidence
│   └── Vikhr-Llama3.1-8B-Instruct-R-21-09-24/
│       ├── validation_50/
│       ├── assessment_261/
│       ├── evidence/
│       └── results/
│
├── Results/                      # Small root-level diagnostics
├── deliverables/                 # Assessment/submission artifacts
├── docs/                         # Human-facing experiment documentation
│
├── SUMMARY.md                    # Engineering summary
├── SILENT_FAILURE_AUDIT.md       # Failure and verification audit
└── SOUP_IMPROVEMENTS.md          # Proposed SOUP improvements
```

Large `checkpoint-*` directories are intentionally excluded from Git. They remain part of the retained training evidence outside the repository.

---

## 📖 Documentation

- **[DPO Experiment Report](deliverables/Report(DPO)Shunya-code.pdf)** — polished two-page report covering training, verification, independent evaluation, and limitations

- **[Experiment Card](docs/EXPERIMENT_CARD.md)** — one-page overview
- **[Reproduction Notes](docs/REPRODUCTION.md)** — environment, data, and verification sequence
- **[Engineering Summary](SUMMARY.md)** — detailed interpretation
- **[Silent Failure Audit](SILENT_FAILURE_AUDIT.md)** — what passed and what failed
- **[SOUP Improvements](SOUP_IMPROVEMENTS.md)** — proposed engineering changes
- **[Deliverable Report](deliverables/DELIVERABLE_REPORT.md)** — submission-oriented audit
- **[Evidence Manifest](deliverables/EVIDENCE_MANIFEST.md)** — evidence inventory
- **[AI Tool Disclosure](deliverables/AI_TOOL_DISCLOSURE.md)** — tooling disclosure

---

## 🧭 Bottom line

This repository does **not** claim that DPO improved the model simply because the loss decreased or a 93% training-set preference score was obtained.

The stronger engineering conclusion is:

> **SOUP successfully executed a DPO/LoRA training run and produced a valid adapter, while the independent evaluation and runtime audit exposed important limits in what can be claimed about generalization and verification.**

That distinction is the point of the project.
