"""Rerun paraphrase experiments with reduced epochs to fit in timeout."""
import modal
import os
import json

app = modal.App("cs224n-paraphrase")

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


@app.function(image=image, gpu="A10G", timeout=14400)
def run_paraphrase(fine_tune_mode: str, lr: float, epochs: int):
    import subprocess
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "paraphrase_detection.py",
        "--fine_tune_mode", fine_tune_mode,
        "--use_gpu",
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch_size", "16",
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
        "para_full": run_paraphrase.spawn("full-model", 1e-5, 5),
        "para_lora": run_paraphrase.spawn("lora", 1e-4, 5),
        "para_last_layer": run_paraphrase.spawn("last-linear-layer", 1e-3, 5),
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
    print("PARAPHRASE RESULTS")
    print("=" * 60)
    for key in ["para_last_layer", "para_full", "para_lora"]:
        if key in results:
            stdout = results[key].get("stdout", "")
            mode = results[key].get("mode", key)
            final_acc = "N/A"
            for line in stdout.split("\n"):
                if "dev paraphrase acc" in line:
                    final_acc = line.split("dev paraphrase acc :: ")[1].strip()
                elif "Epoch" in line and "dev acc" in line:
                    final_acc = line.split("dev acc :: ")[1].strip()
            print(f"  {mode:20s}  Dev Acc = {final_acc}")
    print("=" * 60)

    with open("para_results.json", "w") as f:
        json.dump({k: v.get("stdout", "")[:2000] for k, v in results.items()}, f, indent=2)
