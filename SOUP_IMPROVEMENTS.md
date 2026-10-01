# Proposed SOUP Improvements

## Core observation

The failure exposed by this experiment is not primarily a lack of
training functionality.

SOUP already simplifies training configuration, model loading,
quantization, LoRA setup, checkpointing and evaluation.

The missing layer is **evidence-driven verification**.

A model can have:

- a successful process exit;
- an improving loss;
- checkpoints;
- non-zero adapter weights;

and still fail when someone actually attempts to load and run it.

SOUP should therefore treat training as an observable lifecycle rather
than a single command.

## 1. Preflight hardware feasibility

Before training, SOUP should estimate:

- model parameter memory;
- quantization memory;
- LoRA trainable memory;
- optimizer memory;
- activation memory;
- expected sequence-length cost;
- available GPU VRAM;
- CPU RAM;
- disk/offload requirements.

The result should explicitly say whether the requested configuration is
expected to fit.

Example:

    PRELIGHT CHECK
    Model: 8B
    Quantization: 4-bit
    GPU VRAM: 14.6 GB
    Estimated training requirement: ...
    Status: FIT / RISK / DOES NOT FIT

## 2. Resolve and record configuration

The exact effective configuration should be written into the run
directory before training starts.

This should include:

- model identifier;
- model revision/hash when available;
- dataset identifier/hash;
- tokenizer identifier;
- quantization configuration;
- LoRA configuration;
- sequence length;
- batch size;
- gradient accumulation;
- optimizer;
- scheduler;
- learning rate;
- number of epochs;
- random seeds;
- software versions;
- hardware information.

A run should never depend on reconstructing this information later.

## 3. Parameter-update verification

SOUP should verify that intended trainable parameters actually change.

At minimum:

    parameter_before
    parameter_after
    update_norm
    nonzero_update_count

This catches cases where training appears to run but the intended
parameters are not being updated.

## 4. Event-based training telemetry

Instead of relying primarily on terminal text, emit structured events:

    run_started
    preflight_completed
    model_loaded
    dataset_loaded
    trainer_created
    optimizer_created
    first_backward
    first_parameter_update
    checkpoint_created
    evaluation_started
    evaluation_completed
    runtime_test_started
    runtime_test_completed
    run_completed
    run_failed

Each event should contain timestamps and relevant metadata.

This makes post-run forensic analysis much easier.

## 5. Fresh-process verification

The most important proposed change is to test the artifact independently
from the process that trained it.

After training:

1. terminate or isolate the training model;
2. load the saved adapter from disk;
3. load the intended base model;
4. verify device placement;
5. generate a deterministic test response;
6. record success/failure.

This catches stale Python objects, broken offload state, missing files and
device-placement problems.

## 6. Device/offload sanity checks

SOUP should explicitly detect:

- meta tensors;
- CPU tensors unexpectedly involved in GPU execution;
- disk-offloaded modules;
- inconsistent device maps;
- missing Accelerate hooks;
- incompatible offload indices.

If generation cannot execute safely, the run should report:

    ARTIFACT CREATED
    RUNTIME VERIFICATION FAILED

rather than simply:

    TRAINING COMPLETE

## 7. Independent hardware telemetry

Reported speed and VRAM should be distinguished from independently
measured telemetry.

SOUP should periodically record:

- GPU memory allocated;
- GPU memory reserved;
- GPU utilization;
- temperature;
- power;
- iteration timing.

The report should identify whether a value is measured independently
or simply reported by the training framework.

## 8. Evaluation integrity

DPO evaluation should record:

- dataset hash;
- example count;
- train/evaluation split;
- duplicate detection;
- overlap detection where practical;
- exact evaluation configuration;
- base and adapted model identities.

A 93% preference score should be presented as an evaluation result,
not as proof that training was valid.

## 9. Actionable failure diagnosis

When verification fails, SOUP should explain the next action.

For example:

    RUNTIME VERIFICATION FAILED

    Cause:
    Model contains CPU/disk-offloaded modules incompatible with the
    current generation path.

    Recommended actions:
    1. Restart runtime.
    2. Free GPU memory.
    3. Reload using 4-bit quantization.
    4. Use a compatible device map.
    5. Retry fresh-process verification.

This is much more useful than a generic stack trace.

## 10. Final evidence grade

The final report should separate:

    TRAINING EXECUTION
    ARTIFACT INTEGRITY
    PARAMETER UPDATE
    RUNTIME EXECUTION
    EVALUATION
    HARDWARE TELEMETRY

A run should not receive one undifferentiated PASS merely because the
training process exited successfully.

## Proposed principle

SOUP should make the simplest path the safest path:

    configure
       ↓
    preflight
       ↓
    train
       ↓
    checkpoint
       ↓
    verify artifact
       ↓
    fresh-process runtime test
       ↓
    evaluate
       ↓
    explain evidence

The goal is not to replace the trainer.

The goal is to make it difficult for a training run to appear successful
when the resulting artifact has not actually been verified.
