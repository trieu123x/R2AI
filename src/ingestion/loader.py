import re

with open('retriever.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Inject functions before def retrieve
functions_code = '''
    def expand_query(self, query: str) -> str:
        """Query Expansion bằng rule-based dictionary."""
        q_lower = query.lower()
        expanded_terms = []
        
        intents = {
            "hỗ trợ": ["ưu đãi", "chính sách hỗ trợ", "khuyến khích"],
            "ưu đãi": ["miễn", "giảm", "hỗ trợ"],
            "miễn": ["giảm", "không phải nộp", "ưu đãi"],
            "giảm": ["miễn", "ưu đãi thuế"],
            "đất đai": ["tiền thuê đất", "tiền sử dụng đất", "mặt bằng", "đất"],
            "thuế": ["thuế thu nhập doanh nghiệp", "thuế tndn", "miễn thuế", "giảm thuế"]
        }
        
        for k, v in intents.items():
            if k in q_lower:
                expanded_terms.extend(v)
                
        if expanded_terms:
            new_terms = [t for t in expanded_terms if t not in q_lower]
            if new_terms:
                return query + " " + " ".join(new_terms)
        return query

    def expand_to_parent_article(self, results: list) -> list:
        """Parent Retrieval: Lấy toàn bộ nội dung của Điều luật từ các chunk ban đầu."""
        if not results:
            return []
            
        target_articles = {}
        for r in results:
            if not r.article_hint:
                continue
            doc_id = r.document_id
            if doc_id not in target_articles:
                target_articles[doc_id] = set()
            target_articles[doc_id].add(r.article_hint)
            
        if not target_articles:
            return results
            
        doc_ids = list(target_articles.keys())
        conn = self._get_conn()
        cur = conn.cursor()
        
        placeholders = ",".join("?" for _ in doc_ids)
        cur.execute(f"""
            SELECT document_id, chunk_index, content
            FROM document_chunks
            WHERE document_id IN ({placeholders})
            ORDER BY document_id, chunk_index
        """, doc_ids)
        
        all_chunks = cur.fetchall()
        
        doc_chunks = {}
        for row in all_chunks:
            did = row["document_id"]
            if did not in doc_chunks:
                doc_chunks[did] = []
            doc_chunks[did].append(row)
            
        expanded_articles = {}
        
        for did, rows in doc_chunks.items():
            current_article = None
            for row in rows:
                content = row["content"] or ""
                # Rely on global extract_article_hint
                hint = extract_article_hint(content)
                if hint:
                    current_article = hint
                
                if current_article and current_article in target_articles[did]:
                    key = (did, current_article)
                    if key not in expanded_articles:
                        expanded_articles[key] = []
                    expanded_articles[key].append(content)
                    
        final_results = []
        seen_keys = set()
        
        for r in results:
            if not r.article_hint:
                final_results.append(r)
                continue
                
            key = (r.document_id, r.article_hint)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            if key in expanded_articles:
                full_content = "\\n...\\n".join(expanded_articles[key])
                r.content = full_content
                r.source += "_parent_expanded"
                final_results.append(r)
            else:
                final_results.append(r)
                
        return final_results

    def retrieve(
'''

content = content.replace('    def retrieve(', functions_code)

# 2. Modify retrieve to use expand_query and expand_to_parent_article
retrieve_start = '''    def retrieve(
        self,
        query: str,
        mode: Literal["vector", "fts", "hybrid"] = "fts",
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        t0 = time.time()
        k = top_k or self.top_k
        
        # 1. Query Expansion (Intent Understanding)
        expanded_query = self.expand_query(query)
        
        # 2. Truy xuất danh sách ứng viên (candidate pool)
        if rerank:
            fetch_k = 50
        else:
            fetch_k = max(50, k * 4)
        
        if mode == "vector":
            results = self.vector_search(expanded_query, top_k=fetch_k)
        elif mode == "fts":
            results = self.fts_search(expanded_query, top_k=fetch_k)
        else:
            results = self.hybrid_search(expanded_query, top_k=fetch_k)
            
        # 3. Tiến hành lọc trùng lặp trước để loại bỏ nhiễu
        unique_results = []
        for r in results:
            is_dup = False
            for ur in unique_results:
                if self.are_chunks_duplicate(r, ur):
                     is_dup = True
                     break
            if not is_dup:
                unique_results.append(r)
                
        # 4. Parent Retrieval (Thay thế cho aggregate_chunks_into_articles)
        article_results = self.expand_to_parent_article(unique_results)
'''

# Find the block from `    def retrieve(` to `        # 3. Rerank và lọc`
# and replace it.
pattern = re.compile(r'    def retrieve\(.*?# 3\. Rerank và lọc', re.DOTALL)
content = pattern.sub(retrieve_start + '\n        # 5. Rerank và lọc', content)

with open('retriever.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Update local_retriever.py
content = content.replace('class LegalRetriever:', 'class LocalRetriever:')
content = content.replace('LegalRetriever(', 'LocalRetriever(')

with open('local_retriever.py', 'w', encoding='utf-8') as f:
    f.write(content)
