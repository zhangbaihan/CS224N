"""Download predictions from Modal Volume."""
import modal
import os

app = modal.App("cs224n-download")
vol = modal.Volume.from_name("cs224n-outputs")


@app.function(volumes={"/outputs": vol})
def download_outputs():
    vol.reload()
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
    if not files:
        print("Volume is empty - tasks may not have completed yet.")
