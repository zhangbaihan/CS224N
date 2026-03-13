"""
Run improved CS224N experiments on Modal GPUs.
All improvements: LR warmup+decay, gradient clipping, weight decay,
fixed prompt mismatch, data augmentation.
"""
import modal
import os
import json

app = modal.App("cs224n-improved")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "tqdm==4.58.0", "requests==2.25.1", "filelock==3.0.12",
        "tokenizers==0.20", "einops==0.8.0", "transformers==4.46.3",
        "scikit-learn", "importlib-metadata==3.7.0", "sacrebleu==2.5.1",
    )
    .add_local_dir(
        local_path=".",
        remote_path="/root/project",
        ignore=["*.pt", ".git", "__pycache__", "predictions",
                "*.zip"],
    )
)


@app.function(image=image, gpu="A10G", timeout=14400)
def run_classifier(fine_tune_mode: str, lr: float, epochs: int, batch_size: int = 32):
    """Train improved SST + CFIMDB classifier."""
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "improved/classifier.py",
        "--fine-tune-mode", fine_tune_mode,
        "--use_gpu",
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"classifier failed: {result.stderr[-2000:]}")

    return {"stdout": result.stdout, "mode": fine_tune_mode}


@app.function(image=image, gpu="A10G", timeout=14400)
def run_paraphrase(fine_tune_mode: str, lr: float, epochs: int,
                   batch_size: int = 8, model_size: str = "gpt2"):
    """Train improved paraphrase detection."""
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "improved/paraphrase_detection.py",
        "--fine_tune_mode", fine_tune_mode,
        "--use_gpu",
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--model_size", model_size,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"paraphrase failed: {result.stderr[-2000:]}")

    return {"stdout": result.stdout, "mode": fine_tune_mode}


@app.local_entrypoint()
def main():
    experiments = {
        # Classifier: full-model with higher LR, more epochs, bigger batch
        "clf_full_2e5": run_classifier.spawn("full-model", 2e-5, 20, 32),
        "clf_full_5e5": run_classifier.spawn("full-model", 5e-5, 20, 32),
        "clf_lora": run_classifier.spawn("lora", 1e-4, 20, 32),

        # Paraphrase: fixed prompts + augmentation + schedule
        "para_full_1e5": run_paraphrase.spawn("full-model", 1e-5, 10, 8),
        "para_full_2e5": run_paraphrase.spawn("full-model", 2e-5, 10, 8),
        "para_lora": run_paraphrase.spawn("lora", 1e-4, 10, 8),
    }

    results = {}
    for name, future in experiments.items():
        try:
            result = future.get()
            results[name] = result
            print(f"\n{'='*60}")
            print(f"COMPLETED: {name} (mode={result['mode']})")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\nFAILED: {name}: {e}")
            results[name] = {"stdout": str(e), "mode": name}

    # Summary
    print("\n\n" + "=" * 80)
    print("IMPROVED RESULTS SUMMARY")
    print("=" * 80)

    print("\n--- SST & CFIMDB (Sentiment Classification) ---")
    for key in ["clf_full_2e5", "clf_full_5e5", "clf_lora"]:
        if key in results and "stdout" in results[key]:
            stdout = results[key]["stdout"]
            mode = results[key].get("mode", key)
            # Parse last dev acc for SST and CFIMDB
            sst_acc, cfimdb_acc = "N/A", "N/A"
            current = None
            for line in stdout.split("\n"):
                if "Training Sentiment Classifier on SST" in line:
                    current = "sst"
                elif "Training Sentiment Classifier on cfimdb" in line:
                    current = "cfimdb"
                elif "Epoch" in line and "dev acc" in line:
                    acc = line.split("dev acc :: ")[1].strip() if "dev acc :: " in line else "?"
                    if current == "sst":
                        sst_acc = acc
                    elif current == "cfimdb":
                        cfimdb_acc = acc
                elif line.strip().startswith("dev acc ::"):
                    acc = line.split("dev acc :: ")[1].strip()
                    if current == "sst":
                        sst_acc = acc
                    elif current == "cfimdb":
                        cfimdb_acc = acc
            print(f"  {key:20s} ({mode})  SST={sst_acc}  CFIMDB={cfimdb_acc}")

    print("\n--- Quora Paraphrase Detection ---")
    for key in ["para_full_1e5", "para_full_2e5", "para_lora"]:
        if key in results and "stdout" in results[key]:
            stdout = results[key]["stdout"]
            mode = results[key].get("mode", key)
            final_acc = "N/A"
            for line in stdout.split("\n"):
                if "dev paraphrase acc" in line:
                    final_acc = line.split("dev paraphrase acc :: ")[1].strip()
                elif "Epoch" in line and "dev acc" in line:
                    final_acc = line.split("dev acc :: ")[1].strip()
            print(f"  {key:20s} ({mode})  Dev Acc = {final_acc}")

    print("\n" + "=" * 80)

    with open("improved_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[:3000] for k, v in results.items()}, f, indent=2)
    print("Full logs saved to improved_results.json")
