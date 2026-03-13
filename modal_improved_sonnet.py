"""Run improved sonnet generation on Modal."""
import modal
import os
import json

app = modal.App("cs224n-improved-sonnet")

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


@app.function(image=image, gpu="A10G", timeout=7200)
def run_sonnet(fine_tune_mode: str, lr: float, epochs: int, batch_size: int = 8):
    """Train improved sonnet generation."""
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "improved/sonnet_generation.py",
        "--fine_tune_mode", fine_tune_mode,
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

    return {"stdout": result.stdout, "mode": fine_tune_mode, "lr": lr}


@app.local_entrypoint()
def main():
    experiments = {
        "sonnet_full_1e5": run_sonnet.spawn("full-model", 1e-5, 10, 8),
        "sonnet_full_5e5": run_sonnet.spawn("full-model", 5e-5, 10, 8),
        "sonnet_lora": run_sonnet.spawn("lora", 1e-4, 10, 8),
    }

    results = {}
    for name, future in experiments.items():
        try:
            result = future.get()
            results[name] = result
            print(f"\nCOMPLETED: {name}")
        except Exception as e:
            print(f"\nFAILED: {name}: {e}")
            results[name] = {"stdout": str(e), "mode": name}

    print("\n" + "=" * 60)
    print("IMPROVED SONNET RESULTS")
    print("=" * 60)
    for key, res in results.items():
        stdout = res.get("stdout", "")
        mode = res.get("mode", key)
        lr = res.get("lr", "?")
        # Find last train loss and CHRF score
        last_loss, chrf = "N/A", "N/A"
        for line in stdout.split("\n"):
            if "Epoch" in line and "train loss" in line:
                last_loss = line.split("train loss :: ")[1].strip().rstrip(".")
            if "Dev CHRF score" in line:
                chrf = line.split("Dev CHRF score :: ")[1].strip()
        print(f"  {key:25s} ({mode}, lr={lr})  Loss={last_loss}  Dev CHRF={chrf}")
    print("=" * 60)

    with open("improved_sonnet_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[-3000:] for k, v in results.items()}, f, indent=2)
    print("Saved to improved_sonnet_results.json")
