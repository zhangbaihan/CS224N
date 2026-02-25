import modal
import os, sys

app = modal.App("paraphrase-detection")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "tqdm", "transformers", "tokenizers", 
                 "scikit-learn", "sacrebleu", "requests", "importlib_metadata",
                 "einops")
    .add_local_dir(".", remote_path="/root/project", ignore=["*.pt", "*.pth", "__pycache__"])
)

@app.function(image=image, gpu="A10G", timeout=60 * 60 * 20)
def train_remote():
    project_dir = "/root/project"
    os.chdir(project_dir)
    sys.path.insert(0, project_dir)

    import paraphrase_detection as pd

    args = pd.get_args()
    args.use_gpu = True
    args.epochs = 1
    args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'

    pd.seed_everything(args.seed)
    pd.train(args)
    pd.test(args)

    with open(args.para_dev_out, "rb") as f:
        dev_bytes = f.read()

    with open(args.para_test_out, "rb") as f:
        test_bytes = f.read()

    return {
        "dev": dev_bytes,
        "test": test_bytes
    }

@app.local_entrypoint()
def main():
    outputs = train_remote.remote()

    with open("predictions/para-dev-output.csv", "wb") as f:
        f.write(outputs["dev"])

    with open("predictions/para-test-output.csv", "wb") as f:
        f.write(outputs["test"])

    print("Saved prediction files locally in predictions/")