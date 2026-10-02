# Engineering Summary — SOUP Russian DPO

## Executive summary

This repository documents a Russian-language DPO + LoRA training run using SOUP and the `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24` base model on a Tesla T4.

The experiment provides strong evidence that training executed and produced a real LoRA adapter. The independent evaluation, however, does **not** support a simple claim that DPO improved the model overall.

The strongest result is the verification story: training evidence is substantial, and checkpoint-level parameter updates are now directly verified. Runtime and independent-comparison gaps remain documented rather than overstated.

## 1. Run configuration

- Base model: `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24`
- Method: DPO + LoRA
- Quantization: 4-bit
- Supplied training data: 500 examples
- SOUP training split: 450 train + 50 validation
- Independent assessment: 261 held-out test examples
- Epochs: 3
- Steps: 339
- Learning rate: `5e-6`
- Batch size: 1
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `v_proj`
- Hardware: Tesla T4, approximately 14.6 GiB

## 2. Evidence that training occurred

The retained artifacts include:

- timestamped training log;
- checkpoints at steps 200, 300 and 339;
- trainer state and learning-rate history;
- changing loss values;
- LoRA configuration;
- final adapter with 128 tensors and 6,815,744 finite, non-zero parameters;
- successful LoRA runtime generation on 5/5 prompts.

Recorded loss moved from approximately 0.6928 to 0.1326 at checkpoint 200, 0.0907 at checkpoint 300, and 0.0708 at checkpoint 339.

These artifacts strongly support the conclusion that a training process executed and produced a changing LoRA artifact.

## 3. Evaluation results

### 50-example validation

- Base: 43/50 = 86.0%
- LoRA: 45/50 = 90.0%
- Change: +4.0 percentage points

This validation set is sampled from the supplied training split and therefore is useful as a validation/control measurement, not as independent held-out evidence.

### 261-example independent assessment

- Base: 229/261 = 87.739%
- LoRA: 228/261 = 87.356%
- Change: -0.383 percentage points
- Mean margin: 0.16850 → 0.16983
- Mean-margin change: +0.00132

Transitions:

- Base correct → LoRA correct: 218
- Base wrong → LoRA wrong: 22
- Base correct → LoRA wrong: 11
- Base wrong → LoRA correct: 10

The adapter therefore recovered 10 previously wrong examples while losing 11 previously correct examples on this assessment.

### Prompted-base control

On the same 261-example assessment:

- Base: 87.739%
- Prompted base: 89.272%
- Change: +1.533 percentage points
- Mean margin: 0.16850 → 0.19967

This control is important because it shows that prompt formatting alone affected the measured preference score.

### 93/100 DPO evaluation

A separate 100-example DPO preference evaluation produced:

- 93 chosen > rejected
- 93% preference accuracy
- mean margin: +0.887668
- median margin: +0.765336

That evaluation uses `Data/train.jsonl`. It is therefore treated as **training-set verification**, not held-out generalization evidence.

## 4. Runtime verification failure

The original base-model runtime check failed because the existing base model had an incompatible CPU/disk offload state. The recorded error was:

`Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

The LoRA model generated successfully on all five runtime prompts.

A clean base-vs-LoRA generation comparison was therefore not valid in that runtime. This is a verification-environment failure, not evidence that the adapter itself is bad.

## 5. Independent integrity checks performed

The repository audit also verified:

- train JSONL: 500 valid records;
- test JSONL: 261 valid records;
- validation JSONL: 50 valid records;
- assessment JSONL: 261 valid records;
- train/test exact-record overlap: 0;
- train/test prompt overlap: 0;
- assessment set exactly corresponds to the held-out test split;
- validation-50 is intentionally sampled from the training split with seed 42;
- duplicate legacy result files were byte-for-byte identical to their canonical copies and were consolidated.

## 6. What remains unproven

The following should not be inferred from the current evidence:

- that every intended optimizer update occurred beyond the checkpoint-level parameter-change evidence;
- that the dataset's preference labels are semantically correct;
- that there is no deeper semantic leakage beyond exact/prompt overlap;
- that the independent assessment is statistically improved by DPO;
- that the original base model can be cleanly compared in the recorded runtime;
- that the reported framework memory values equal an independently measured peak;
- that the training-state argument preservation was complete.

## 7. Parameter-update verification

The checkpoint-level update check is now complete. Comparing checkpoint-200 with checkpoint-339 produced:

- 128 / 128 LoRA tensors changed
- 6,815,680 / 6,815,744 adapter elements changed
- changed fraction: 99.999061%
- mean absolute delta: 5.51219836e-05
- maximum absolute delta: 1.68222934e-04
- result: **PASS**

The raw output is preserved at `Evaluations/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/results/parameter_update_verification.txt`. The large checkpoint directories remain intentionally excluded from Git.

## 8. Engineering conclusion

The defensible conclusion is not simply **"DPO improved the model."**

It is:

> **SOUP successfully executed a DPO/LoRA training run and produced a valid adapter. Independent assessment was essentially flat, while the audit exposed concrete runtime and verification gaps that should be addressed before claiming end-to-end correctness.**

That distinction is the central engineering result of this project.
