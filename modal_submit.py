"""
Train paraphrase + sonnet and download predictions for submission.
"""
import modal
import os

app = modal.App("cs224n-submit2")
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
def run_paraphrase():
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
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"paraphrase failed: {result.stderr[-2000:]}")

    os.makedirs("/outputs/predictions", exist_ok=True)
    for f in os.listdir("predictions"):
        if "para" in f:
            shutil.copy2(f"predictions/{f}", f"/outputs/predictions/{f}")
            print(f"Saved predictions/{f}")
    vol.commit()
    return {"stdout": result.stdout}


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/outputs": vol})
def run_sonnet():
    import subprocess, shutil
    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python", "improved/sonnet_generation.py",
        "--fine_tune_mode", "full-model",
        "--use_gpu",
        "--lr", "1e-5",
        "--epochs", "10",
        "--batch_size", "8",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"sonnet failed: {result.stderr[-2000:]}")

    os.makedirs("/outputs/predictions", exist_ok=True)
    for f in os.listdir("predictions"):
        if "sonnet" in f:
            shutil.copy2(f"predictions/{f}", f"/outputs/predictions/{f}")
            print(f"Saved predictions/{f}")
    vol.commit()
    return {"stdout": result.stdout}


@app.function(volumes={"/outputs": vol})
def download_outputs():
    results = []
    for root, dirs, files in os.walk("/outputs"):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, "/outputs")
            size = os.path.getsize(full)
            entry = {"path": rel, "size": size}
            if rel.startswith("predictions/"):
                with open(full, "r") as fh:
                    entry["content"] = fh.read()
            results.append(entry)
    return results


@app.local_entrypoint()
def main():
    para_future = run_paraphrase.spawn()
    sonnet_future = run_sonnet.spawn()

    for name, future in [("paraphrase", para_future), ("sonnet", sonnet_future)]:
        try:
            result = future.get()
            stdout = result.get("stdout", "")
            for line in stdout.split("\n"):
                if any(k in line for k in ["dev acc", "dev paraphrase", "CHRF", "Epoch"]):
                    print(f"[{name}] {line.strip()}")
            print(f"\nCOMPLETED: {name}")
        except Exception as e:
            print(f"\nFAILED: {name}: {e}")

    print("\n" + "=" * 60)
    print("DOWNLOADING PREDICTIONS...")
    print("=" * 60)
    files = download_outputs.remote()
    os.makedirs("predictions", exist_ok=True)
    for f in files:
        print(f"  {f['path']} ({f['size']} bytes)")
        if "content" in f:
            local_path = f['path']
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as fh:
                fh.write(f["content"])
            print(f"    -> saved locally to {local_path}")

    print("\n" + "=" * 60)
    print("DONE! Run `python prepare_submit.py` to create submission zip.")
    print("=" * 60)
