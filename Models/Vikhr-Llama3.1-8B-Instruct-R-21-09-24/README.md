---
base_model: Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24
library_name: peft
tags:
- dpo
- lora
- transformers
- trl
pipeline_tag: text-generation
---

# Vikhr Russian DPO LoRA Adapter

This directory contains the LoRA adapter produced by the SOUP Russian DPO experiment.

## Base model

`Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24`

## Training

- Method: DPO
- Quantization: 4-bit
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `v_proj`
- Epochs: 3
- Steps: 339
- Learning rate: `5e-6`
- Batch size: 1

## Adapter contents

The committed adapter contains:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- chat template
- `training_args.bin`
- model-specific `soup.yaml`

Large training checkpoints are intentionally not stored in Git.

## Load the adapter

The adapter is a PEFT artifact and must be loaded on top of the matching base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24"
adapter_path = "Models/Vikhr-Llama3.1-8B-Instruct-R-21-09-24"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, adapter_path)
```

For the original low-VRAM training environment, use the same compatible quantization/device-loading strategy recorded in the experiment documentation rather than blindly loading a second full 8B model.

## Evaluation note

The adapter successfully generated 5/5 runtime test responses in the recorded verification run. The independent 261-example assessment moved from 87.74% for the base condition to 87.36% for the LoRA condition, so this artifact should not be presented as having demonstrated overall held-out improvement.

See the repository [Experiment Card](../../docs/EXPERIMENT_CARD.md) and [Engineering Summary](../../SUMMARY.md).

## Framework versions

- PEFT 0.20.0
- TRL 0.24.0
- Transformers 4.57.6
- PyTorch 2.11.0+cu128
- Datasets 4.8.5
- Tokenizers 0.22.2
