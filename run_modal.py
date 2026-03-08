import modal
import os

app = modal.App("cs224n-sonnet")

vol = modal.Volume.from_name("cs224n-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "tqdm==4.58.0",
        "requests==2.25.1",
        "filelock==3.0.12",
        "tokenizers==0.20",
        "einops==0.8.0",
        "transformers==4.46.3",
        "scikit-learn",
        "importlib-metadata==3.7.0",
    )
    .add_local_dir(
        local_path="/Users/baihanzhang/Desktop/CS224N/CS224N",
        remote_path="/root/project",
        ignore=["*.pt", ".git", "__pycache__", "predictions", "run_modal.py"],
    )
)


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": vol})
def train_sonnets():
    import subprocess

    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    result = subprocess.run(
        ["python", "sonnet_generation.py", "--use_gpu"],
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Copy checkpoints and predictions to the volume
    for f in os.listdir("/root/project"):
        if f.endswith(".pt"):
            os.system(f"cp /root/project/{f} /vol/{f}")
            print(f"Saved {f} to volume")

    sonnet_path = "/root/project/predictions/generated_sonnets.txt"
    if os.path.exists(sonnet_path):
        os.system("cp /root/project/predictions/generated_sonnets.txt /vol/generated_sonnets.txt")
        with open(sonnet_path) as f:
            content = f.read()
        print("=== Generated Sonnets ===")
        print(content)
        return content

    vol.commit()
    return ""


@app.local_entrypoint()
def main():
    sonnets = train_sonnets.remote()
    out_dir = "/Users/baihanzhang/Desktop/CS224N/CS224N/predictions"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/generated_sonnets.txt", "w") as f:
        f.write(sonnets)
    print(f"Saved generated_sonnets.txt ({len(sonnets)} chars)")
    print("\nCheckpoints saved to Modal volume 'cs224n-checkpoints'.")
    print("To download them: modal volume get cs224n-checkpoints <filename> .")
