# Reproduction Notes

## Environment

The recorded training environment used:

- Python 3.13.15
- PyTorch 2.11.0 + CUDA 12.8
- Transformers 4.57.6
- TRL 0.24.0
- PEFT 0.20.0
- Accelerate 1.14.0
- bitsandbytes 0.50.2
- SOUP CLI 0.72.4
- Tesla T4, approximately 14.6 GiB VRAM

## Configuration

The model-specific configuration is stored at:

`Models/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/soup.yaml`

It uses 4-bit quantization, 3 epochs, learning rate `5e-6`, batch size `1`, LoRA rank `16`, and alpha `32`.

## Data layout

- `Data/train.jsonl`: 500 examples supplied to SOUP
- `Data/test.jsonl`: 261 held-out assessment examples
- SOUP internally used 450 training examples and 50 validation examples from the supplied 500.

The repository's independent checks found no exact-record or prompt overlap between train and test.

## Verification ladder

For a future rerun, verify in this order:

1. Preflight hardware and package versions.
2. Validate dataset structure and split boundaries.
3. Run SOUP training.
4. Preserve timestamped logs and every produced checkpoint.
5. Verify checkpoint integrity.
6. Compare adapter tensors between checkpoint-200 and checkpoint-339 with `deliverables/verify_parameter_updates.py`. The recorded run is preserved at `Evaluations/Vikhr-Llama3.1-8B-Instruct-R-21-09-24/results/parameter_update_verification.txt` and reports PASS.
7. Start a fresh process and test model loading.
8. Run base-vs-LoRA generation with independently loaded models.
9. Run the 50-example validation and 261-example assessment.
10. Preserve raw GPU telemetry and resolved configuration.

The checkpoint directories remain intentionally excluded from Git, but the completed verification result is preserved as raw text in the evaluation results. To reproduce it independently, run the script against the retained checkpoint-200 and checkpoint-339 directories on Google Drive.
