import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.query_rewrite import QueryRewriter
from src.retrieval.pipeline_retriever import PipelineRetriever
from src.retrieval.reranker import PipelineReranker
from src.utils.context_compressor import ContextCompressor
from src.llm.generator import PipelineGenerator
from src.utils.validator import PipelineValidator

class LegalRAGPipeline:
    """
    Toàn bộ RAG Pipeline chuẩn theo các bước:
    1: Question
    2: Query Rewrite
    3: BM25 (top 10) + Embedding (top 10)
    4: Merge
    5: Reranker
    6: Top 5
    7: Context Compression
    8: Qwen3-8B
    9: Citation Validation
    10: Final Answer
    """
    def __init__(self, use_llm_rewrite=False, db_path=None, llm_model_name="Qwen/Qwen3-8B-Instruct"):
        print("[pipeline] Khởi tạo các module...")
        self.generator = PipelineGenerator(model_name=llm_model_name)
        
        # Step 2
        self.rewriter = QueryRewriter(use_llm=use_llm_rewrite, llm_generator=self.generator)
        
        # Step 3 & 4
        self.retriever = PipelineRetriever(db_path=db_path, top_k_each=10)
        
        # Step 5 & 6
        self.reranker = PipelineReranker(top_n=5)
        
        # Step 7
        self.compressor = ContextCompressor()
        
        # Step 9
        self.validator = PipelineValidator()
        
        print("[pipeline] Khởi tạo hoàn tất.")

    def _generate_rule_based_answer(self, results) -> str:
        """Fallback: Tạo câu trả lời rule-based nếu LLM lỗi liên tục."""
        if not results:
            return "Không tìm thấy căn cứ pháp lý liên quan để trả lời câu hỏi này."
        
        answer_parts = ["1. Trả lời trực tiếp: Căn cứ vào các văn bản pháp luật hiện hành, dưới đây là thông tin trích xuất:"]
        analysis = []
        cơ_sở = []
        
        for idx, r in enumerate(results[:3], start=1):
            snippet = r.content.strip()
            if "Nội dung:" in snippet:
                snippet = snippet.split("Nội dung:", 1)[1].strip()
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            analysis.append(f"- Căn cứ {idx} quy định: {snippet}")
            cơ_sở.append(f"- {r.article_hint or 'Quy định'} {r.legal_type} số {r.doc_number}")

        answer_parts.append("2. Phân tích chi tiết:\n" + "\n".join(analysis))
        answer_parts.append("3. Căn cứ pháp lý:\n" + "\n".join(cơ_sở))
        answer_parts.append("4. Hạn chế của dữ liệu (nếu có): Dựa trên trích xuất tự động.")
        return "\n\n".join(answer_parts)

    def run(self, question: str, max_retries: int = 2) -> dict:
        """
        Thực thi pipeline.
        Trả về dictionary chứa: query, rewritten_query, results (Top 5), context, answer, is_valid.
        """
        # Step 1: Question
        query = question.strip()
        if not query:
            return {"answer": "Câu hỏi trống."}

        # Step 2: Query Rewrite
        rewrite_res = self.rewriter.rewrite(query)
        rewritten_query = rewrite_res["rewritten_query"]

        # Step 3 & 4: Retrieval & Merge
        # Chúng ta dùng rewritten_query để search nhằm tăng độ bao phủ
        merged_results = self.retriever.retrieve_and_merge(rewritten_query)

        # Step 5 & 6: Reranker & Top 5
        # Reranker nên đánh giá độ liên quan với original_query để đảm bảo chính xác ngữ nghĩa gốc
        top5_results = self.reranker.rerank_and_filter(query, merged_results)

        # Step 7: Context Compression
        compressed_context = self.compressor.compress(top5_results)

        # Step 8, 9, 10: Generator, Validator, Final Answer
        final_answer = ""
        is_valid = False
        warning_msg = None
        
        for attempt in range(max_retries):
            # Step 8: Qwen3-8B
            answer = self.generator.generate_answer(
                query=query, 
                context=compressed_context, 
                warning_msg=warning_msg
            )
            
            # Step 9: Citation Validation
            valid, err_msg = self.validator.validate(answer, top5_results)
            if valid:
                final_answer = answer
                is_valid = True
                break
            else:
                print(f"[pipeline] Phát hiện lỗi sinh ({err_msg}) lần {attempt+1}. Đang sinh lại...")
                warning_msg = f"LƯU Ý LỚN: Ở lượt sinh trước, bạn đã mắc lỗi: {err_msg}. Hãy chú ý sửa lỗi này, không được lặp lại và không dẫn chiếu sai lệch."
        
        # Nếu vẫn không hợp lệ sau max_retries, fallback sang rule-based
        if not is_valid:
            print("[pipeline] Đã thử tối đa số lần, fallback sang Rule-based.")
            final_answer = self._generate_rule_based_answer(top5_results)

        # Step 10: Final Answer (trả về cùng metadata)
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "top5_results": top5_results,
            "compressed_context": compressed_context,
            "final_answer": final_answer,
            "is_valid": is_valid
        }
