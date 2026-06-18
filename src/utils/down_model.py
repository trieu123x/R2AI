from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="./bge-reranker-v2-m3"
)
