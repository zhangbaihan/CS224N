"""
Retry LoRA alpha sweep for paraphrase and sonnet only.
Classifier alpha sweep already completed (alpha had no effect: 0.501 SST, 0.963 CFIMDB for all).
Keeps lr=1e-4, r=8 fixed. Tests alpha=8, 32 (alpha=16 is the default from prior runs).
"""
import modal
import os
import json

app = modal.App("cs224n-lora-alpha-retry")

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
def run_paraphrase_lora(lora_alpha: float):
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)
    cmd = [
        "python", "improved/paraphrase_detection.py",
        "--fine_tune_mode", "lora",
        "--use_gpu",
        "--lr", "1e-4",
        "--epochs", "5",
        "--batch_size", "16",
        "--lora_alpha", str(lora_alpha),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"paraphrase failed: {result.stderr[-2000:]}")
    return {"stdout": result.stdout, "alpha": lora_alpha, "task": "para"}


@app.function(image=image, gpu="A10G", timeout=7200)
def run_sonnet_lora(lora_alpha: float):
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)
    cmd = [
        "python", "improved/sonnet_generation.py",
        "--fine_tune_mode", "lora",
        "--use_gpu",
        "--lr", "1e-4",
        "--epochs", "10",
        "--batch_size", "8",
        "--lora_alpha", str(lora_alpha),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"sonnet failed: {result.stderr[-2000:]}")
    return {"stdout": result.stdout, "alpha": lora_alpha, "task": "sonnet"}


@app.local_entrypoint()
def main():
    experiments = {
        "para_alpha8": run_paraphrase_lora.spawn(8.0),
        "para_alpha32": run_paraphrase_lora.spawn(32.0),
        "sonnet_alpha8": run_sonnet_lora.spawn(8.0),
        "sonnet_alpha32": run_sonnet_lora.spawn(32.0),
    }

    results = {}
    for name, future in experiments.items():
        try:
            result = future.get()
            results[name] = result
            print(f"\nCOMPLETED: {name}")
        except Exception as e:
            print(f"\nFAILED: {name}: {e}")
            results[name] = {"stdout": str(e), "alpha": name, "task": name}

    print("\n" + "=" * 70)
    print("LORA ALPHA SWEEP RESULTS — PARAPHRASE & SONNET (lr=1e-4, r=8)")
    print("=" * 70)

    for name, res in sorted(results.items()):
        stdout = res.get("stdout", "")
        alpha = res.get("alpha", "?")

        para_acc, chrf = "N/A", "N/A"
        for line in stdout.split("\n"):
            if "dev paraphrase acc" in line:
                para_acc = line.split("dev paraphrase acc :: ")[1].strip()
            elif "Epoch" in line and "dev acc" in line:
                para_acc = line.split("dev acc :: ")[1].strip() if "dev acc :: " in line else "?"
            elif "Dev CHRF score" in line:
                chrf = line.split("Dev CHRF score :: ")[1].strip()

        if "para" in name:
            print(f"  {name:25s} (alpha={alpha})  Para={para_acc}")
        elif "sonnet" in name:
            print(f"  {name:25s} (alpha={alpha})  CHRF={chrf}")

    print("=" * 70)

    with open("lora_alpha_retry_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[-3000:] for k, v in results.items()}, f, indent=2)
    print("Saved to lora_alpha_retry_results.json")
