"""
LoRA learning rate sweep across all 3 tasks.
Tests lr=1e-4, 3e-4, 5e-4 for LoRA on SST, paraphrase, and sonnet.
"""
import modal
import os
import json

app = modal.App("cs224n-lora-sweep")

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
        ignore=["*.pt", ".git", "__pycache__", "predictions", "*.zip"],
    )
)


@app.function(image=image, gpu="A10G", timeout=14400)
def run_classifier_lora(lr: float, epochs: int = 20, batch_size: int = 32):
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)
    cmd = [
        "python", "improved/classifier.py",
        "--fine-tune-mode", "lora",
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
    return {"stdout": result.stdout, "lr": lr, "task": "clf"}


@app.function(image=image, gpu="A10G", timeout=14400)
def run_paraphrase_lora(lr: float, epochs: int = 5, batch_size: int = 16):
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)
    cmd = [
        "python", "improved/paraphrase_detection.py",
        "--fine_tune_mode", "lora",
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
        raise RuntimeError(f"paraphrase failed: {result.stderr[-2000:]}")
    return {"stdout": result.stdout, "lr": lr, "task": "para"}


@app.function(image=image, gpu="A10G", timeout=7200)
def run_sonnet_lora(lr: float, epochs: int = 10, batch_size: int = 8):
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)
    cmd = [
        "python", "improved/sonnet_generation.py",
        "--fine_tune_mode", "lora",
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
        raise RuntimeError(f"sonnet failed: {result.stderr[-2000:]}")
    return {"stdout": result.stdout, "lr": lr, "task": "sonnet"}


@app.local_entrypoint()
def main():
    lrs = [1e-4, 3e-4, 5e-4]

    experiments = {}
    for lr in lrs:
        lr_tag = f"{lr:.0e}".replace("+", "").replace("-0", "-")
        experiments[f"clf_lora_{lr_tag}"] = run_classifier_lora.spawn(lr)
        experiments[f"para_lora_{lr_tag}"] = run_paraphrase_lora.spawn(lr)
        experiments[f"sonnet_lora_{lr_tag}"] = run_sonnet_lora.spawn(lr)

    results = {}
    for name, future in experiments.items():
        try:
            result = future.get()
            results[name] = result
            print(f"\nCOMPLETED: {name}")
        except Exception as e:
            print(f"\nFAILED: {name}: {e}")
            results[name] = {"stdout": str(e), "lr": name, "task": name}

    # Parse and display results
    print("\n" + "=" * 70)
    print("LORA LR SWEEP RESULTS")
    print("=" * 70)

    for name, res in sorted(results.items()):
        stdout = res.get("stdout", "")
        lr = res.get("lr", "?")

        # Parse SST dev acc
        sst_acc, cfimdb_acc, para_acc, chrf = "N/A", "N/A", "N/A", "N/A"
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
                else:
                    para_acc = acc
            elif "dev paraphrase acc" in line:
                para_acc = line.split("dev paraphrase acc :: ")[1].strip()
            elif "Dev CHRF score" in line:
                chrf = line.split("Dev CHRF score :: ")[1].strip()

        if "clf" in name:
            print(f"  {name:30s} SST={sst_acc}  CFIMDB={cfimdb_acc}")
        elif "para" in name:
            print(f"  {name:30s} Para={para_acc}")
        elif "sonnet" in name:
            print(f"  {name:30s} CHRF={chrf}")

    print("=" * 70)

    with open("lora_sweep_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[-3000:] for k, v in results.items()}, f, indent=2)
    print("Saved to lora_sweep_results.json")
