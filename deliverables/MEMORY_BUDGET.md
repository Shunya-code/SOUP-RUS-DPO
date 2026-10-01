# Memory Budget — Russian DPO Run

## Known configuration

- Model: `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24`
- Approximate parameter count: 8B
- Quantization: 4-bit
- DPO, batch size 1
- LoRA r=16, alpha=32
- Trainable adapter parameters observed: 6,815,744
- GPU: Tesla T4
- Reported GPU memory: ~14.6 GB

## Lower-bound weight estimate

4-bit raw storage:

`8,000,000,000 × 0.5 bytes = 4,000,000,000 bytes`

`4,000,000,000 / 2^30 ≈ 3.73 GiB`

This is only the raw 4-bit parameter payload. Quantization scales/metadata, framework buffers and allocator overhead are additional.

## Adapter/optimizer estimate

6,815,744 LoRA parameters:

- FP16/BF16 adapter weights: ~13.0 MiB
- FP16/BF16 gradients: ~13.0 MiB
- Two FP32 Adam moments: ~52.0 MiB
- Combined adapter + gradients + two FP32 moments: ~78.0 MiB

These are estimates, not measured allocations.

## Activations and logits

The exact activation footprint cannot be reconstructed from the committed evidence because the resolved effective sequence length and detailed model hidden dimensions/retention behaviour for this run are not preserved in the evidence.

For DPO, chosen and rejected examples both participate in the preference computation. A naive full-logit estimate would be:

`rows × sequence_length × vocabulary_size × bytes_per_logit`

but this is **not** a reliable measurement of the actual TRL/SOUP peak because preference-loss implementations can reduce logits to token log-probabilities rather than retain a full FP32 vocabulary tensor.

## Reference-model pass

A second resident 8B reference model would be roughly another quantized model-sized allocation plus overhead, but this run's evidence does not establish that two complete resident copies were used. Therefore no second-model VRAM number is asserted.

## Actual peak VRAM

**Not available in the committed repository evidence.**

The repo records approximately 14.6 GB total T4 memory and records that additional full-model loading caused OOM, but it does not preserve a raw independent peak `nvidia-smi` trace. The submission should not substitute total card capacity for peak training allocation.

## Gap

Because actual peak allocated/reserved VRAM is missing, the requested estimate-vs-measurement gap cannot honestly be calculated yet.

### Evidence required

Capture:

```bash
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

and, preferably during training/evaluation, periodic telemetry or a process-level memory trace. Also record:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

after resetting peak statistics at the beginning of the measured interval.
