# SOUP Russian DPO — Submission Report

## Verdict: DON'T SHIP

The evidence strongly supports that a real DPO/LoRA training run executed, but it does not satisfy the stronger requirement that the resulting artifact and hardware/runtime path have been independently verified end-to-end.

### 1. Memory budget before training

Run configuration: `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24`, DPO, 4-bit, LoRA r=16/alpha=32, batch size 1, 3 epochs, 339 steps, Tesla T4 with approximately 14.6 GB reported VRAM.

A first-order lower-bound estimate for the 8B base weights is:

- 8,000,000,000 parameters × 0.5 bytes/parameter for 4-bit storage
- = 4.00 GB decimal ≈ 3.73 GiB raw weight storage
- Quantization metadata/scales add overhead, so this is not the complete resident model footprint.

Adapter state from the audit: 6,815,744 parameters.

- FP16/BF16 LoRA weights: 6,815,744 × 2 bytes ≈ 13.0 MiB.
- Adam-style first + second moments, if FP32: ≈ 52.0 MiB.
- FP16/BF16 gradients: ≈ 13.0 MiB.
- Adapter + gradients + two FP32 optimizer moments: ≈ 78 MiB before allocator/framework overhead.

DPO also evaluates chosen and rejected responses, so activation/token memory depends strongly on the effective sequence length and vocabulary. The repository does not preserve enough resolved sequence-length/activation telemetry to reconstruct an exact activation peak. The reference-model pass also cannot be assigned a second full-model VRAM cost without knowing the actual SOUP loading/streaming path used in this run.

**Actual peak VRAM:** the repository documents only approximately 14.6 GB total T4 memory. I did not find an independent raw `nvidia-smi` peak trace in the committed evidence. Therefore I do **not** invent a peak number. The gap between the estimate and actual peak is consequently not quantifiable from the committed evidence.

### 2. Proof that the model trained

The evidence is materially stronger than “loss went down”:

- Training completed with return code 0.
- Checkpoints exist at steps 200, 300 and 339.
- Trainer state reports global steps 200, 300 and 339.
- Loss changed from 0.6928 to 0.1326 at checkpoint 200, 0.0907 at 300 and 0.0708 at 339.
- Learning-rate history exists.
- The final adapter contains 128 tensors and 6,815,744 finite, non-zero parameters.
- LoRA runtime generation succeeded on 5/5 prompts.
- Independent DPO evaluation gave 93/100 chosen-over-rejected.

A direct before/after checkpoint comparison has now been completed. Checkpoint-200 → checkpoint-339 changed all 128 LoRA tensors and 6,815,680 of 6,815,744 adapter elements (99.999061%). Mean absolute delta was 5.51219836e-05 and maximum absolute delta was 1.68222934e-04. The raw verification output is preserved at `Evaluations/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/results/parameter_update_verification.txt`. This materially strengthens the evidence that the LoRA adapter was updated during training.

What this check would **not** detect: a wrong dataset, label/masking bug, evaluation leakage, incorrect base-model revision, or an adapter that changed but learned the wrong thing.

### 3. Silent failures

| Failure mode | Evidence/check | SOUP checks catch it? | Conclusion |
|---|---|---|---|
| Training command exits but intended adapter is not updated | Checkpoints, non-zero adapter; checkpoint-200 → 339 tensor delta | Yes, for the checkpoint comparison | PASS — 128/128 tensors changed |
| Wrong/offloaded runtime device placement | Base generation failed with CUDA/CPU mismatch | Existing artifact checks did not catch it | Confirmed failure |
| Second full 8B load exceeds VRAM | Reload attempt produced OOM | Runtime verification exposed it | Confirmed limitation |
| Dataset schema/row count problem | JSONL validation; 500 train / 261 test expected | Partly | Structural validity supported |
| Dataset semantic error / bad preference pairs | No semantic adjudication evidence | No | Open risk |
| Train/eval leakage | No committed overlap/leakage proof | No | Open risk |
| Incorrect optimizer updates/gradients | No per-update gradient/update telemetry | No | Open risk |
| Hardware telemetry is only framework-reported | No raw independent peak trace committed | No | Open risk |
| Base-vs-LoRA generation comparison | Base runtime failed, comparison skipped | No | Unresolved |
| Trainer-state argument preservation | 12/13 forensic checks; this one failed | No | Known evidence gap |

The existing audit explicitly records the base-model failure: `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`. The LoRA model generated 5/5, but that does not repair the missing base comparison.

### 4. Required changes before SHIP

2. Capture and commit the raw `nvidia-smi` output from the training/evaluation environment, including memory usage.
3. Record the resolved effective sequence length and the actual peak allocated/reserved VRAM.
4. Run the requested Soup `data doctor` / pre-flight / `soup ship` checks on the exact artifact and preserve raw outputs.
5. Perform fresh-process artifact loading and a clean base-vs-LoRA runtime comparison after resetting/freeing the GPU.
6. Record dataset overlap/leakage checks and the exact dataset/config/model revisions used for evaluation.

The overall evidence package remains **NOT FULLY VERIFIED FOR SHIP** because the base-vs-LoRA comparison and independent hardware telemetry gaps remain. The parameter-update requirement itself is now **PASS**.

### AI-tool disclosure

AI assistance was used to inspect the repository, identify missing evidence, reconstruct the deliverable structure, and draft/check the verification logic. Repository claims were checked against the committed source material. Missing measurements were not fabricated; they are explicitly marked as missing and converted into concrete evidence requirements.
