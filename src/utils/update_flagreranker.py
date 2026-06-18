import re

with open('retriever.py', 'r', encoding='utf-8') as f:
    content = f.read()

rerank_new = '''    def rerank(self, query: str, results: List[RetrievalResult], top_k: Optional[int] = None) -> List[RetrievalResult]:
        if not results:
            return []
            
        if self._reranker is None:
            import torch
            from FlagEmbedding import FlagReranker
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[local] Loading reranker model BAAI/bge-reranker-v2-m3 using FlagEmbedding on device '{device}'...", flush=True)
            t0 = time.time()
            # Sử dụng FlagReranker thay cho CrossEncoder
            self._reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
            print(f"[local] Reranker model loaded in {time.time()-t0:.1f}s", flush=True)
            
        pairs = []
        for r in results:
            best_text = getattr(r, 'best_chunk_content', None) or r.content
            
            # Thêm metadata boosting để tăng độ chính xác (rất hiệu quả với văn bản pháp luật)
            article_str = f"Điều: {r.article_hint}\\n" if r.article_hint else ""
            doc_text = f"Văn bản: {r.legal_type} {r.doc_number} - {r.title}\\n{article_str}Nội dung:\\n{best_text}"
            
            pairs.append([query, doc_text])
            
        t_rerank = time.time()
        # Compute scores with FlagReranker
        scores = self._reranker.compute_score(pairs, normalize=True)
        # Nếu có 1 cặp thì nó trả về float, nếu nhiều cặp trả về list
        if isinstance(scores, float):
            scores = [scores]
            
        print(f"[local] Reranking took {time.time() - t_rerank:.2f}s for {len(pairs)} pairs.")
        
        LEGAL_WEIGHT = {'''

# Find the block to replace
pattern = re.compile(r'    def rerank\(self, query: str, results: List\[RetrievalResult\], top_k: Optional\[int\] = None\) -> List\[RetrievalResult\]:.*?LEGAL_WEIGHT = \{', re.DOTALL)
content = pattern.sub(rerank_new, content)

with open('retriever.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Update local_retriever.py
content = content.replace('class LegalRetriever:', 'class LocalRetriever:')
content = content.replace('LegalRetriever(', 'LocalRetriever(')

with open('local_retriever.py', 'w', encoding='utf-8') as f:
    f.write(content)
