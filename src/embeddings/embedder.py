import os
import time
import numpy as np

class VietnameseBiEncoder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[embedder] Loading bi-encoder on device '{device}'...", flush=True)
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name, device=device)
            print(f"[embedder] Model loaded in {time.time()-t0:.1f}s", flush=True)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([query], show_progress_bar=False)[0]
        return vec.astype(np.float32)
