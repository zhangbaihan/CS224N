"""Run just the 2 missing paraphrase LoRA alpha experiments (alpha=8, 32)."""
import modal
import os
import json

app = modal.App("cs224n-para-alpha")

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
        print("STDERR:", result.stderr[-1000:])
    if result.returncode != 0:
        raise RuntimeError(f"paraphrase failed: {result.stderr[-2000:]}")
    return {"stdout": result.stdout, "alpha": lora_alpha}


@app.local_entrypoint()
def main():
    f8 = run_paraphrase_lora.spawn(8.0)
    f32 = run_paraphrase_lora.spawn(32.0)

    for name, future in [("alpha=8", f8), ("alpha=32", f32)]:
        try:
            result = future.get()
            stdout = result["stdout"]
            # Extract dev acc
            for line in stdout.split("\n"):
                if "dev paraphrase acc" in line:
                    print(f"RESULT {name}: {line.strip()}")
            print(f"COMPLETED: {name}")
        except Exception as e:
            print(f"FAILED: {name}: {e}")

    print("Done.")
