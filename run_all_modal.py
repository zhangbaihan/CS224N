"""
Run all CS224N experiments on Modal GPUs.

Tasks: SST, CFIMDB, Paraphrase, Sonnet
Modes: last-linear-layer, full-model, lora
"""
import modal
import os
import json

app = modal.App("cs224n-all-experiments")

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
        ignore=["*.pt", ".git", "__pycache__", "predictions", "run_*.py",
                "*.zip", "modal_run*.py"],
    )
)


@app.function(image=image, gpu="T4", timeout=7200)
def run_classifier(fine_tune_mode: str, lr: float, epochs: int):
    """Train SST + CFIMDB classifier with given mode."""
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "classifier.py",
        "--fine-tune-mode", fine_tune_mode,
        "--use_gpu",
        "--lr", str(lr),
        "--epochs", str(epochs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"classifier.py failed: {result.stderr[-2000:]}")

    output = result.stdout
    return {"stdout": output, "mode": fine_tune_mode}


@app.function(image=image, gpu="T4", timeout=7200)
def run_paraphrase(fine_tune_mode: str, lr: float, epochs: int):
    """Train paraphrase detection with given mode."""
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "paraphrase_detection.py",
        "--fine_tune_mode", fine_tune_mode,
        "--use_gpu",
        "--lr", str(lr),
        "--epochs", str(epochs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"paraphrase_detection.py failed: {result.stderr[-2000:]}")

    return {"stdout": result.stdout, "mode": fine_tune_mode}


@app.function(image=image, gpu="T4", timeout=7200)
def run_sonnet(fine_tune_mode: str, lr: float, epochs: int):
    """Train sonnet generation with given mode."""
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "sonnet_generation.py",
        "--fine_tune_mode", fine_tune_mode,
        "--use_gpu",
        "--lr", str(lr),
        "--epochs", str(epochs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"sonnet_generation.py failed: {result.stderr[-2000:]}")

    return {"stdout": result.stdout, "mode": fine_tune_mode}


def parse_classifier_metrics(stdout):
    """Extract dev acc for SST and CFIMDB from classifier output."""
    metrics = {}
    lines = stdout.split("\n")
    current_dataset = None
    for line in lines:
        if "Training Sentiment Classifier on SST" in line:
            current_dataset = "sst"
        elif "Training Sentiment Classifier on cfimdb" in line:
            current_dataset = "cfimdb"
        elif "Epoch" in line and "dev acc" in line and current_dataset:
            parts = line.split("dev acc :: ")
            if len(parts) > 1:
                acc = parts[1].strip()
                metrics[f"{current_dataset}_dev_acc"] = acc
        elif line.strip().startswith("dev acc ::") and current_dataset:
            acc = line.split("dev acc :: ")[1].strip()
            metrics[f"{current_dataset}_final_dev_acc"] = acc
    return metrics


def parse_paraphrase_metrics(stdout):
    """Extract dev acc from paraphrase output."""
    metrics = {}
    lines = stdout.split("\n")
    for line in lines:
        if "Epoch" in line and "dev acc" in line:
            parts = line.split("dev acc :: ")
            if len(parts) > 1:
                metrics["last_epoch_dev_acc"] = parts[1].strip()
        if "dev paraphrase acc" in line:
            parts = line.split("dev paraphrase acc :: ")
            if len(parts) > 1:
                metrics["final_dev_acc"] = parts[1].strip()
    return metrics


def parse_sonnet_metrics(stdout):
    """Extract final train loss from sonnet output."""
    metrics = {}
    lines = stdout.split("\n")
    for line in lines:
        if "Epoch" in line and "train loss" in line:
            parts = line.split("train loss :: ")
            if len(parts) > 1:
                metrics["last_train_loss"] = parts[1].strip().rstrip(".")
    return metrics


@app.local_entrypoint()
def main():
    experiments = {
        "classifier_last_layer": run_classifier.spawn("last-linear-layer", 1e-3, 10),
        "classifier_full": run_classifier.spawn("full-model", 1e-5, 10),
        "classifier_lora": run_classifier.spawn("lora", 1e-4, 10),
        "para_full": run_paraphrase.spawn("full-model", 1e-5, 10),
        "para_lora": run_paraphrase.spawn("lora", 1e-4, 10),
        "para_last_layer": run_paraphrase.spawn("last-linear-layer", 1e-3, 10),
        "sonnet_full": run_sonnet.spawn("full-model", 1e-5, 10),
        "sonnet_lora": run_sonnet.spawn("lora", 1e-4, 10),
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

    print("\n\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n--- SST & CFIMDB (Sentiment Classification) ---")
    for key in ["classifier_last_layer", "classifier_full", "classifier_lora"]:
        if key in results and "stdout" in results[key]:
            m = parse_classifier_metrics(results[key]["stdout"])
            mode = results[key]["mode"]
            sst = m.get("sst_final_dev_acc", m.get("sst_dev_acc", "N/A"))
            cfimdb = m.get("cfimdb_final_dev_acc", m.get("cfimdb_dev_acc", "N/A"))
            print(f"  {mode:20s}  SST={sst}  CFIMDB={cfimdb}")

    print("\n--- Quora Paraphrase Detection ---")
    for key in ["para_last_layer", "para_full", "para_lora"]:
        if key in results and "stdout" in results[key]:
            m = parse_paraphrase_metrics(results[key]["stdout"])
            mode = results[key]["mode"]
            acc = m.get("final_dev_acc", m.get("last_epoch_dev_acc", "N/A"))
            print(f"  {mode:20s}  Dev Acc={acc}")

    print("\n--- Sonnet Generation ---")
    for key in ["sonnet_full", "sonnet_lora"]:
        if key in results and "stdout" in results[key]:
            m = parse_sonnet_metrics(results[key]["stdout"])
            mode = results[key]["mode"]
            loss = m.get("last_train_loss", "N/A")
            print(f"  {mode:20s}  Final Train Loss={loss}")

    print("\n" + "=" * 80)

    with open("experiment_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[:500] for k, v in results.items()}, f, indent=2)
    print("Full logs saved to experiment_results.json")
