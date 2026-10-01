"""
SOUP — Russian DPO / LoRA Training + Assessment
================================================

Cleaned workflow reconstructed from the original Colab assessment.

KEPT:
- Google Drive dataset paths
- DPO dataset validation
- SOUP training configuration / YAML generation
- Preflight checks
- SOUP dry-run
- Timestamped `soup train` execution
- Checkpoint/artifact verification
- Independent DPO evaluation
- Machine-readable evaluation results

REMOVED:
- Duplicate evaluation implementations
- Prompted-base comparison
- Failed CPU/disk-offloaded base generation path
- Diagnostic generation branches that did not produce reliable results
- Hard-coded fallback metrics
- GitHub/repository-generation noise
- Repeated forensic/report-generation copies

IMPORTANT:
- TRAINING IS NOT STARTED automatically unless RUN_TRAINING=True.
- Evaluation uses checkpoint-339 by default.
- The failed base-model generation branch is intentionally not repeated.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

BASE_DIR = Path("/content/drive/MyDrive/Soup/RussianDPO")
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
EVALUATION_DIR = BASE_DIR / "evaluation" / "results"

TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
YAML_PATH = BASE_DIR / "soup.yaml"
TRAIN_LOG_PATH = BASE_DIR / "training_timestamped.log"

CHECKPOINT_STEP = 339
CHECKPOINT_DIR = OUTPUT_DIR / f"checkpoint-{CHECKPOINT_STEP}"

# Set True only when you intentionally want to launch a new training run.
RUN_TRAINING = False

# Existing experiment configuration.
BASE_MODEL = "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24"
TASK = "dpo"
LANGUAGE = "ru"
QUANTIZATION = "4bit"

EPOCHS = 3
LEARNING_RATE = "5e-6"
BATCH_SIZE = 1
LORA_R = 16
LORA_ALPHA = 32

# Evaluation size. The independent test set contains 261 examples.
MAX_EVAL_EXAMPLES = 100
SEED = 42

EXPECTED_TRAIN_ROWS = 500
EXPECTED_TEST_ROWS = 261
REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


# ============================================================================
# 1. GOOGLE DRIVE / DIRECTORY SETUP
# ============================================================================

def prepare_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOUP — RUSSIAN DPO / LORA WORKFLOW")
    print("=" * 78)
    print(f"Base directory : {BASE_DIR}")
    print(f"Train dataset  : {TRAIN_PATH}")
    print(f"Test dataset   : {TEST_PATH}")
    print(f"Output         : {OUTPUT_DIR}")
    print(f"YAML           : {YAML_PATH}")
    print()


# ============================================================================
# 2. DATASET VALIDATION
# ============================================================================

def validate_jsonl(path: Path, expected_columns: tuple[str, ...]) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {path}"
                ) from exc

            missing = [column for column in expected_columns if column not in item]
            if missing:
                raise ValueError(
                    f"{path}: line {line_number} is missing {missing}"
                )

            for column in expected_columns:
                if not isinstance(item[column], str):
                    raise ValueError(
                        f"{path}: line {line_number}, "
                        f"'{column}' must be a string"
                    )

            rows += 1

    return rows


def validate_datasets() -> tuple[int, int]:
    print("=" * 78)
    print("DATASET VALIDATION")
    print("=" * 78)

    train_count = validate_jsonl(TRAIN_PATH, REQUIRED_COLUMNS)
    test_count = validate_jsonl(TEST_PATH, REQUIRED_COLUMNS)

    print(f"Train examples : {train_count}")
    print(f"Test examples  : {test_count}")

    if train_count != EXPECTED_TRAIN_ROWS:
        print(
            f"[WARN] Expected {EXPECTED_TRAIN_ROWS} train examples, "
            f"found {train_count}."
        )

    if test_count != EXPECTED_TEST_ROWS:
        print(
            f"[WARN] Expected {EXPECTED_TEST_ROWS} test examples, "
            f"found {test_count}."
        )

    print("[PASS] DPO JSONL structure is valid.")
    print()
    return train_count, test_count


# ============================================================================
# 3. SOUP YAML GENERATION
# ============================================================================

def generate_soup_yaml() -> str:
    """
    Generate the training configuration used by the original SOUP run.

    The YAML is written to Google Drive so it survives a Colab runtime
    disconnect.
    """

    yaml_text = f"""base: {BASE_MODEL}
task: {TASK}

data:
  train: {TRAIN_PATH}
  test: {TEST_PATH}

training:
  epochs: {EPOCHS}
  lr: {LEARNING_RATE}
  batch_size: {BATCH_SIZE}
  lora:
    r: {LORA_R}
    alpha: {LORA_ALPHA}
  quantization: {QUANTIZATION}
  output: {OUTPUT_DIR}
"""

    YAML_PATH.write_text(yaml_text.strip() + "\n", encoding="utf-8")

    print("=" * 78)
    print("GENERATED SOUP YAML")
    print("=" * 78)
    print(YAML_PATH)
    print("-" * 78)
    print(YAML_PATH.read_text(encoding="utf-8"))
    return yaml_text


# ============================================================================
# 4. HARDWARE / PRE-FLIGHT CHECKS
# ============================================================================

def preflight_checks() -> bool:
    print("=" * 78)
    print("SOUP PRE-FLIGHT CHECK")
    print("=" * 78)

    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed.")
        return False

    checks: dict[str, bool] = {
        "Google Drive": Path("/content/drive/MyDrive").exists(),
        "Train dataset": TRAIN_PATH.exists(),
        "Test dataset": TEST_PATH.exists(),
        "YAML": YAML_PATH.exists(),
        "Output directory": OUTPUT_DIR.exists(),
        "CUDA": torch.cuda.is_available(),
    }

    for name, passed in checks.items():
        print(f"{'[PASS]' if passed else '[FAIL]'} {name}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA: {torch.version.cuda}")

        try:
            x = torch.randn(512, 512, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            del x, y
            torch.cuda.empty_cache()
            print("[PASS] PyTorch CUDA operation")
        except Exception as exc:
            print(f"[FAIL] PyTorch CUDA operation: {exc}")
            checks["CUDA operation"] = False
        else:
            checks["CUDA operation"] = True
    else:
        checks["CUDA operation"] = False

    all_passed = all(checks.values())
    print()
    print(
        "[PASS] All preflight checks passed."
        if all_passed
        else "[WARN] Preflight requires attention."
    )
    print()

    return all_passed


# ============================================================================
# 5. SOUP CLI CHECK / DRY RUN
# ============================================================================

def run_soup_help_and_dry_run() -> None:
    print("=" * 78)
    print("SOUP CLI CHECK")
    print("=" * 78)

    subprocess.run(["soup", "train", "--help"], check=False)

    print()
    print("=" * 78)
    print("SOUP DRY RUN")
    print("=" * 78)

    subprocess.run(
        [
            "soup",
            "train",
            "-c",
            str(YAML_PATH),
            "--dry-run",
        ],
        check=False,
    )


# ============================================================================
# 6. TRAINING
# ============================================================================

def run_training() -> int:
    """
    Run the actual SOUP training command and stream a timestamped log
    to Google Drive.

    This is intentionally opt-in through RUN_TRAINING.
    """

    command = [
        "soup",
        "train",
        "--config",
        str(YAML_PATH),
        "--name",
        "russian-dpo-vikhr",
        "-y",
    ]

    start = datetime.now()

    print("=" * 78)
    print("SOUP TRAINING")
    print("=" * 78)
    print("START:", start.isoformat())
    print("COMMAND:", " ".join(command))
    print("LOG:", TRAIN_LOG_PATH)
    print()

    with TRAIN_LOG_PATH.open("w", encoding="utf-8") as log:
        log.write(f"TRAINING START: {start.isoformat()}\n")
        log.write(f"COMMAND: {' '.join(command)}\n\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            timestamp = datetime.now().isoformat()
            output = f"[{timestamp}] {line}"
            print(output, end="")
            log.write(output)
            log.flush()

        return_code = process.wait()

        end = datetime.now()
        log.write(f"\nTRAINING END: {end.isoformat()}\n")
        log.write(f"RETURN CODE: {return_code}\n")

    print()
    print("=" * 78)
    print(f"Training return code: {return_code}")
    print(f"Training log: {TRAIN_LOG_PATH}")
    print("=" * 78)

    return return_code


# ============================================================================
# 7. CHECKPOINT / ARTIFACT VERIFICATION
# ============================================================================

def verify_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> dict[str, Any]:
    print("=" * 78)
    print("CHECKPOINT VERIFICATION")
    print("=" * 78)
    print("Checkpoint:", checkpoint_dir)

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
    ]

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_dir),
        "files": {},
    }

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")

    for filename in required_files:
        path = checkpoint_dir / filename
        exists = path.exists()
        result["files"][filename] = exists
        print(f"{'[PASS]' if exists else '[FAIL]'} {filename}")

    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(
            adapter_config_path.read_text(encoding="utf-8")
        )

        result["adapter_config"] = adapter_config

        print()
        print("LoRA configuration:")
        for key in (
            "peft_type",
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "inference_mode",
            "base_model_name_or_path",
        ):
            print(f"  {key}: {adapter_config.get(key)}")

    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        trainer_state = json.loads(
            trainer_state_path.read_text(encoding="utf-8")
        )

        result["trainer_state"] = {
            "global_step": trainer_state.get("global_step"),
            "epoch": trainer_state.get("epoch"),
            "best_metric": trainer_state.get("best_metric"),
            "best_model_checkpoint": trainer_state.get(
                "best_model_checkpoint"
            ),
        }

        print()
        print("Trainer state:")
        print("  global_step:", trainer_state.get("global_step"))
        print("  epoch:", trainer_state.get("epoch"))

    print()
    return result


# ============================================================================
# 8. TRAINING CHECKPOINT PROGRESSION
# ============================================================================

def inspect_checkpoint_progression() -> list[dict[str, Any]]:
    print("=" * 78)
    print("CHECKPOINT PROGRESSION")
    print("=" * 78)

    states: list[dict[str, Any]] = []

    for step in (200, 300, 339):
        path = OUTPUT_DIR / f"checkpoint-{step}" / "trainer_state.json"

        if not path.exists():
            print(f"[MISSING] checkpoint-{step}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))

        state = {
            "checkpoint": step,
            "global_step": data.get("global_step"),
            "epoch": data.get("epoch"),
        }

        states.append(state)

        print(
            f"checkpoint-{step}: "
            f"global_step={state['global_step']}, "
            f"epoch={state['epoch']}"
        )

    return states


# ============================================================================
# 9. DPO EVALUATION
# ============================================================================

def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device,
) -> float:
    """
    Compute mean token log-probability of response conditioned on prompt.

    Only response tokens contribute to the score.
    """

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)

    full_ids = tokenizer(
        prompt + response,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2,
            labels.unsqueeze(-1),
        ).squeeze(-1)

    response_start = prompt_ids.shape[1] - 1
    response_token_scores = token_log_probs[:, response_start:]

    return float(response_token_scores.mean().item())


def evaluate_dpo_checkpoint(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    max_examples: int = MAX_EVAL_EXAMPLES,
) -> dict[str, Any]:
    """
    Evaluate chosen-vs-rejected preference using the saved LoRA checkpoint.

    This does NOT perform the failed base-model generation comparison.
    It measures the adapted checkpoint directly on the independent test set.
    """

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("DPO CHECKPOINT EVALUATION")
    print("=" * 78)

    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_config = json.loads(
        adapter_config_path.read_text(encoding="utf-8")
    )

    base_model_name = adapter_config.get(
        "base_model_name_or_path",
        BASE_MODEL,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The original verification used a quantized model on Colab.
    # Keep loading conservative and let Transformers determine the device.
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if QUANTIZATION == "4bit":
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:
            print(f"[WARN] 4-bit configuration unavailable: {exc}")

    print("Base model:", base_model_name)
    print("Adapter:", checkpoint_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
    )

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
    )

    model.eval()

    device = next(
        parameter for parameter in model.parameters()
        if parameter.device.type != "meta"
    ).device

    rows = load_jsonl(TEST_PATH)
    rows = rows[:max_examples]

    results: list[dict[str, Any]] = []

    chosen_better = 0
    margins: list[float] = []

    print(f"Evaluating {len(rows)} examples...")

    for index, row in enumerate(rows):
        chosen_score = sequence_logprob(
            model,
            tokenizer,
            row["prompt"],
            row["chosen"],
            device,
        )

        rejected_score = sequence_logprob(
            model,
            tokenizer,
            row["prompt"],
            row["rejected"],
            device,
        )

        margin = chosen_score - rejected_score
        preferred = margin > 0

        if preferred:
            chosen_better += 1

        margins.append(margin)

        results.append(
            {
                "index": index,
                "prompt": row["prompt"],
                "chosen_logprob": chosen_score,
                "rejected_logprob": rejected_score,
                "margin": margin,
                "chosen_preferred": preferred,
            }
        )

        if (index + 1) % 10 == 0:
            print(f"  evaluated {index + 1}/{len(rows)}")

    accuracy = chosen_better / len(rows) if rows else 0.0
    mean_margin = sum(margins) / len(margins) if margins else 0.0

    sorted_margins = sorted(margins)
    if sorted_margins:
        mid = len(sorted_margins) // 2
        if len(sorted_margins) % 2:
            median_margin = sorted_margins[mid]
        else:
            median_margin = (
                sorted_margins[mid - 1] + sorted_margins[mid]
            ) / 2
    else:
        median_margin = 0.0

    summary = {
        "checkpoint": str(checkpoint_dir),
        "base_model": base_model_name,
        "dataset": str(TEST_PATH),
        "examples": len(rows),
        "chosen_preferred": chosen_better,
        "preference_accuracy": accuracy,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = EVALUATION_DIR / f"clean_checkpoint_{CHECKPOINT_STEP}"
    result_dir.mkdir(parents=True, exist_ok=True)

    json_path = result_dir / "dpo_evaluation.json"
    txt_path = result_dir / "dpo_evaluation.txt"
    per_example_path = result_dir / "per_example_results.json"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    per_example_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt_path.write_text(
        "\n".join(
            [
                "SOUP — CLEAN DPO CHECKPOINT EVALUATION",
                "",
                f"Checkpoint: {checkpoint_dir}",
                f"Base model: {base_model_name}",
                f"Dataset: {TEST_PATH}",
                f"Examples: {len(rows)}",
                f"Chosen > rejected: {chosen_better}/{len(rows)}",
                f"Preference accuracy: {accuracy:.6f}",
                f"Mean margin: {mean_margin:.6f}",
                f"Median margin: {median_margin:.6f}",
                "",
                "Note:",
                "This evaluation measures the saved LoRA checkpoint.",
                "The failed base-vs-LoRA generation branch from the original",
                "workflow is intentionally not rerun here.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("EVALUATION RESULT")
    print("=" * 78)
    print(f"Examples            : {len(rows)}")
    print(f"Chosen > rejected   : {chosen_better}/{len(rows)}")
    print(f"Preference accuracy : {accuracy:.4f}")
    print(f"Mean margin         : {mean_margin:.6f}")
    print(f"Median margin       : {median_margin:.6f}")
    print()
    print("Saved:")
    print(json_path)
    print(txt_path)
    print(per_example_path)

    del model
    del base_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============================================================================
# 10. MAIN WORKFLOW
# ============================================================================

def main() -> None:
    prepare_directories()

    train_count, test_count = validate_datasets()

    # Preserve the original training configuration and YAML generation.
    generate_soup_yaml()

    if not preflight_checks():
        print("Preflight did not fully pass.")
        print("Review the output before training.")
        return

    # The original workflow explicitly inspected SOUP help and dry-ran it.
    run_soup_help_and_dry_run()

    if RUN_TRAINING:
        return_code = run_training()

        if return_code != 0:
            raise RuntimeError(
                f"SOUP training failed with return code {return_code}"
            )

    else:
        print("=" * 78)
        print("TRAINING")
        print("=" * 78)
        print("RUN_TRAINING=False")
        print("Training was NOT started by this script.")
        print()
        print("To intentionally launch training:")
        print("    RUN_TRAINING = True")
        print()

    # Artifact verification is useful whether training was run now
    # or the checkpoint already exists on Drive.
    if CHECKPOINT_DIR.exists():
        verify_checkpoint()
        inspect_checkpoint_progression()

        print()
        print("Checkpoint exists and can be inspected.")
        print(
            "Run evaluate_dpo_checkpoint() explicitly when you want "
            "to load the model and perform GPU evaluation."
        )
    else:
        print()
        print(f"[INFO] {CHECKPOINT_DIR} does not exist yet.")
        print("No evaluation was attempted.")


if __name__ == "__main__":
    main()
