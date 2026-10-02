# Engineering Summary — SOUP Russian DPO Audit

## 1. Objective

The purpose of this work was not simply to determine whether a training
metric improved. The objective was to determine whether a SOUP training
run produced credible evidence of real training and whether the resulting
LoRA artifact could actually be loaded and used.

The experiment used the Russian instruction model
`Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24` with DPO and LoRA on
a Tesla T4 with approximately 14.6 GB of GPU memory.

The configured LoRA adapter used rank 16, alpha 32, dropout 0.05, and
targeted `q_proj` and `v_proj`.

## 2. Evidence that training occurred

The training log records a complete SOUP invocation beginning on
2026-09-30 and ending with return code 0.

The run reports:

- 450 training samples
- 3 epochs
- 339 training steps
- approximately 2 h 26 min duration
- progressive checkpoints at steps 200, 300 and 339

The checkpoints contain trainer state, optimizer state, scheduler state,
RNG state, training arguments, tokenizer information, and the LoRA
adapter.

Trainer state independently shows:

- checkpoint-200 → global step 200
- checkpoint-300 → global step 300
- checkpoint-339 → global step 339

The loss history is not constant:

- initial recorded loss: 0.6928
- checkpoint 200 final recorded loss: 0.1326
- checkpoint 300 final recorded loss: 0.0907
- checkpoint 339 final recorded loss: 0.0708

Learning-rate values were also recorded throughout training.

The final adapter contains 128 tensors and 6,815,744 parameters.
All inspected adapter values are finite and non-zero.

Taken together, these artifacts provide strong evidence that an actual
training process executed and produced a changing LoRA adapter.

## 3. What this evidence does not prove

The audit deliberately distinguishes execution evidence from correctness
evidence.

A changing loss does not prove that the intended dataset was processed
correctly.

Checkpoint existence does not prove that every intended parameter was
updated.

A non-zero LoRA file does not prove that its updates are useful.

Reported GPU memory and speed numbers are not independent hardware
measurements.

The audit also cannot by itself establish that the chosen/rejected
responses were semantically correct or that the evaluation set was free
of leakage.

Most importantly, the original base-model runtime check exposed a
separate problem.

## 4. Silent failure discovered during verification

The existing base model had been loaded using CPU/disk offloading.
Although the object existed and its weights were distributed across
CUDA, CPU and disk, generation failed with:

`Expected all tensors to be on the same device, but found at least two
devices, cuda:0 and cpu`

Attempting to reload the full 8B model without quantization caused GPU
out-of-memory errors. A subsequent 4-bit loading attempt also failed
because the existing Colab GPU was already almost completely occupied.

Rather than repeatedly loading another 8B model, verification reused
the already loaded objects.

The LoRA model successfully generated all five runtime test prompts.

Result:

- LoRA runtime: 5/5
- Base runtime: 0/5
- Base-vs-LoRA generation comparison: skipped

This is an important distinction. The failed base runtime does not prove
that the trained LoRA is bad. It demonstrates that the verification
environment itself contained an execution problem.

## 5. DPO evaluation

An independent DPO evaluation over 100 examples reported:

- chosen > rejected: 93/100
- preference accuracy: 93%
- mean margin: +0.887668
- median margin: +0.765336

These are useful evaluation measurements, but they should not be treated
as proof that the training run itself was valid. They are one layer of
evidence among several.

## 6. What should change in SOUP

The main lesson is not that SOUP needs another training framework.

The more useful direction is an evidence-driven training workflow.

Before training, SOUP should determine whether the selected model,
quantization, sequence length, batch configuration and LoRA configuration
are feasible on the current machine.

During training, SOUP should emit structured events describing model
loading, device placement, trainable parameters, optimizer creation,
gradient/update health, checkpoint creation and hardware telemetry.

After training, SOUP should automatically perform a verification ladder:

1. Artifact integrity.
2. Configuration integrity.
3. Trainable-parameter/update verification.
4. Checkpoint progression.
5. Fresh-process checkpoint loading.
6. Runtime generation.
7. Evaluation integrity.
8. Hardware/resource consistency.

A failed stage should produce an actionable explanation rather than
simply marking the run successful because a loss curve exists.

## 7. Final assessment

The evidence strongly supports that this particular DPO training run
executed and produced a real LoRA artifact.

It does not support the stronger claim that every aspect of the training
and deployment pipeline was correct.

That distinction is the central engineering result of this audit.

The most valuable improvement for SOUP is therefore not another metric.
It is a verification system that understands the relationship between
the selected model, available hardware, training configuration,
checkpoint artifacts and actual runtime behavior.
