"""Train paraphrase and save predictions to Volume. Uses spawn() + detach."""
import modal
import os

app = modal.App("cs224n-para-final")
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


@app.function(image=image, gpu="A10G", timeout=21600, volumes={"/outputs": vol})
def run_and_save():
    import subprocess, shutil
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "improved/paraphrase_detection.py",
        "--fine_tune_mode", "full-model",
        "--use_gpu",
        "--lr", "2e-5",
        "--epochs", "5",
        "--batch_size", "16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"failed: {result.stderr[-2000:]}")

    # Save predictions to volume
    os.makedirs("/outputs/predictions", exist_ok=True)
    for f in os.listdir("predictions"):
        if "para" in f:
            shutil.copy2(f"predictions/{f}", f"/outputs/predictions/{f}")
            print(f"SAVED: predictions/{f}")

    # Write a done marker
    with open("/outputs/para_done.txt", "w") as f:
        f.write(result.stdout[-2000:])

    vol.commit()
    return "done"


@app.local_entrypoint()
def main():
    # Spawn and don't wait — just let it run
    future = run_and_save.spawn()
    print(f"Spawned paraphrase training job.")
    print(f"Check status: modal app list")
    print(f"When done, download with:")
    print(f"  modal volume get cs224n-outputs predictions/para-dev-output.csv predictions/para-dev-output.csv --force")
    print(f"  modal volume get cs224n-outputs predictions/para-test-output.csv predictions/para-test-output.csv --force")
    print(f"  modal volume get cs224n-outputs para_done.txt para_done.txt --force")
