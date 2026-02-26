import os
import modal

app = modal.App("cs224n-sentiment")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "tqdm", "transformers", "tokenizers", 
                 "scikit-learn", "sacrebleu", "requests", "importlib_metadata",
                 "einops")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["*.pt", "*.pth", "__pycache__", "predictions"],
    )
)

SCRIPT_NAME = "classifier.py"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 6)
def run_remote():
    import os
    import subprocess
    import glob

    os.chdir("/root/project")
    os.makedirs("predictions", exist_ok=True)

    cmd = [
        "python",
        SCRIPT_NAME,
        "--fine-tune-mode",
        "full-model",   # change to last-linear-layer if needed
        "--use_gpu",
    ]

    subprocess.check_call(cmd)

    outputs = {}
    for path in glob.glob("predictions/*.csv"):
        with open(path, "rb") as f:
            outputs[path] = f.read()

    return outputs


@app.local_entrypoint()
def main():
    outputs = run_remote.remote()

    for rel_path, data in outputs.items():
        os.makedirs(os.path.dirname(rel_path), exist_ok=True)
        with open(rel_path, "wb") as f:
            f.write(data)