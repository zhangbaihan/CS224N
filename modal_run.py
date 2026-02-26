import modal

app = modal.App("paraphrase-detection")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "tqdm", "transformers", "tokenizers", 
                 "scikit-learn", "sacrebleu", "requests", "importlib_metadata",
                 "einops")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(image=image, gpu="T4", timeout=60 * 60 * 20)
def train_remote():
    import os, sys
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

@app.local_entrypoint()
def main():
    train_remote.remote()