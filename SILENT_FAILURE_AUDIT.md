# Silent-Failure Audit

## Why this audit was performed

A successful training command and an improving loss curve are not
sufficient evidence that the resulting model is usable.

This audit checks multiple independent layers of evidence.

## Evidence that passed

- Timestamped training log exists.
- Training process completed with return code 0.
- Checkpoints exist at steps 200, 300 and 339.
- Checkpoint global steps progress monotonically.
- Trainer state is readable.
- Loss history contains multiple finite measurements.
- Loss is not constant.
- Learning-rate history exists.
- Final LoRA configuration is present.
- LoRA rank and target modules match the intended configuration.
- LoRA adapter contains 6,815,744 non-zero finite parameters.
- LoRA runtime generation succeeded on 5/5 test prompts.
- Independent DPO evaluation produced 93/100 chosen-over-rejected examples.

## Failure discovered

The base model used during verification had an incompatible CPU/disk
offload state.

Attempting generation resulted in:

`Expected all tensors to be on the same device, but found at least two
devices, cuda:0 and cpu`

Attempts to load another full 8B model exceeded the available GPU
memory.

Consequently, a clean base-vs-LoRA generation comparison could not be
performed in the existing runtime.

## Important interpretation

This does not establish that the LoRA model is incorrect.

It establishes that the verification environment contained a runtime
failure that the earlier metric/checkpoint checks did not detect.

That is exactly the class of silent failure that a stronger training
system should surface automatically.

## Remaining verification gaps

The audit does not independently prove:

- semantic correctness of the DPO dataset;
- absence of evaluation leakage;
- non-zero gradients on every training step;
- independent correctness of reported GPU telemetry;
- correctness of every intended optimizer update;
- equivalence between the training-time base revision and every later
  runtime environment.

These should be treated as separate verification layers.
