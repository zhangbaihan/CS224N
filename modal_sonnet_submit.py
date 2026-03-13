"""Train sonnet with best config (lr=5e-5) and save predictions to Volume."""
import modal
import os

app = modal.App("cs224n-sonnet-submit")
vol = modal.Volume.from_name("cs224n-outputs", create_if_missing=True)

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


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/outputs": vol})
def run_and_save():
    import subprocess, shutil
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "improved/sonnet_generation.py",
        "--fine_tune_mode", "full-model",
        "--use_gpu",
        "--lr", "5e-5",
        "--epochs", "10",
        "--batch_size", "8",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"failed: {result.stderr[-2000:]}")

    # Save predictions to volume
    os.makedirs("/outputs/predictions", exist_ok=True)
    for f in os.listdir("predictions"):
        if "sonnet" in f:
            shutil.copy2(f"predictions/{f}", f"/outputs/predictions/{f}")
            print(f"SAVED: predictions/{f}")
    vol.commit()
    return result.stdout


@app.local_entrypoint()
def main():
    print("Training sonnet (full-model, lr=5e-5, 10 epochs)...")
    stdout = run_and_save.remote()

    for line in stdout.split("\n"):
        if any(k in line for k in ["Epoch", "CHRF"]):
            print(line.strip())

    # Download predictions locally
    import subprocess
    os.makedirs("predictions", exist_ok=True)
    for fname in ["generated_sonnets.txt", "generated_sonnets_dev.txt"]:
        r = subprocess.run(
            ["modal", "volume", "get", "cs224n-outputs", f"predictions/{fname}", f"predictions/{fname}"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"Downloaded predictions/{fname}")
        else:
            print(f"Failed: {fname}: {r.stderr}")

    print("\nDONE!")
