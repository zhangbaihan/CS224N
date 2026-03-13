"""Run paraphrase LoRA with batch_size=64 and lr=4e-4, matching teammate's setup."""
import modal
import os
import json

app = modal.App("cs224n-para-lora-bs64")

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
def run_para_lora(lr: float, batch_size: int, epochs: int):
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
    return {"stdout": result.stdout, "lr": lr, "batch_size": batch_size}


@app.local_entrypoint()
def main():
    experiments = {
        # Match teammate: lr=4e-4, bs=64, 10 epochs
        "lora_4e4_bs64": run_para_lora.spawn(4e-4, 64, 10),
        # Also try with our improvements at same bs
        "lora_3e4_bs64": run_para_lora.spawn(3e-4, 64, 10),
        "lora_5e4_bs64": run_para_lora.spawn(5e-4, 64, 10),
    }

    results = {}
    for name, future in experiments.items():
        try:
            result = future.get()
            results[name] = result
            print(f"\nCOMPLETED: {name}")
        except Exception as e:
            print(f"\nFAILED: {name}: {e}")
            results[name] = {"stdout": str(e), "lr": name}

    print("\n" + "=" * 60)
    print("PARAPHRASE LORA BS=64 RESULTS")
    print("=" * 60)
    for name, res in results.items():
        stdout = res.get("stdout", "")
        lr = res.get("lr", "?")
        bs = res.get("batch_size", "?")
        para_acc = "N/A"
        for line in stdout.split("\n"):
            if "dev paraphrase acc" in line:
                para_acc = line.split("dev paraphrase acc :: ")[1].strip()
            elif "Epoch" in line and "dev acc" in line:
                para_acc = line.split("dev acc :: ")[1].strip()
        print(f"  {name:25s} (lr={lr}, bs={bs})  Dev Acc = {para_acc}")
    print("=" * 60)

    with open("lora_bs64_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[-3000:] for k, v in results.items()}, f, indent=2)
    print("Saved to lora_bs64_results.json")
