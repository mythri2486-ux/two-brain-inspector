from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="foersben/mvtec-ad",
    repo_type="dataset",
    allow_patterns=["cable/*", "transistor/*", "metal_nut/*"],
    local_dir="data/mvtec_ad",
)
print("done")