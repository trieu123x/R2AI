"""
test_retrieval.py
=================
CLI tương tác để test pipeline retrieval KHÔNG cần LLM.

Cách dùng:
  python retrieval/test_retrieval.py
  python retrieval/test_retrieval.py --mode fts
  python retrieval/test_retrieval.py --mode vector --top-k 5
  python retrieval/test_retrieval.py --query "Điều kiện thành lập doanh nghiệp" --mode hybrid

Tính năng:
  - 3 chế độ: fts / vector / hybrid
  - Hiển thị nội dung chunk + metadata văn bản
  - Tổng hợp kết quả theo document
  - Format output sẵn để nộp bài thi (relevant_docs, relevant_articles)
  - Benchmark so sánh tốc độ 3 mode
"""

import os
import sys
import io
import re
import time
import json
import argparse
import textwrap

# Force UTF-8 output on Windows to handle Vietnamese text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8", errors="replace")


def _sanitize_query(q: str) -> str:
    """Remove surrogate characters that can appear from encoding issues."""
    return q.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever, RetrievalResult


# ─── ANSI Colors ───────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    GRAY    = "\033[90m"
    WHITE   = "\033[97m"

def colorize(text, color):
    return f"{color}{text}{C.RESET}"

def hr(char="─", width=70, color=C.GRAY):
    return colorize(char * width, color)


# ─── Display helpers ───────────────────────────────────────────────────────────

def print_header():
    print()
    print(colorize("=" * 70, C.CYAN))
    print(colorize("  R2AI Legal Retrieval Test Pipeline  (no LLM mode)", C.CYAN))
    print(colorize("=" * 70, C.CYAN))
    print()


def print_result(idx: int, r: RetrievalResult, show_content: bool = True, max_chars: int = 400):
    mode_color = {
        "vector": C.BLUE,
        "fts":    C.GREEN,
        "hybrid": C.MAGENTA,
    }.get(r.source, C.WHITE)

    print(f"\n{colorize(f'[{idx}]', C.YELLOW)} "
          f"{colorize(r.doc_number, C.BOLD)} "
          f"{colorize(r.legal_type, C.GRAY)}")
    print(f"    {colorize('Tiêu đề:', C.CYAN)} {r.title}")
    print(f"    {colorize('Chunk:', C.CYAN)} #{r.chunk_index}  "
          f"{colorize('Điều:', C.CYAN)} {r.article_hint or '—'}  "
          f"{colorize('Score:', C.CYAN)} {colorize(f'{r.score:.4f}', mode_color)}  "
          f"{colorize('[' + r.source + ']', mode_color)}")

    if show_content and r.content:
        content_preview = r.content[:max_chars]
        if len(r.content) > max_chars:
            content_preview += " ..."
        wrapped = textwrap.fill(content_preview, width=80,
                                initial_indent="    │ ",
                                subsequent_indent="    │ ")
        print(colorize(wrapped, C.GRAY))


def print_submission_format(results: list[RetrievalResult]):
    """In ra định dạng relevant_docs và relevant_articles cho bài nộp."""
    retriever = LegalRetriever.__new__(LegalRetriever)  # chỉ dùng static method
    docs_seen = set()
    articles_seen = set()
    relevant_docs = []
    relevant_articles = []

    for r in results:
        doc_str = r.format_relevant_doc()
        if doc_str not in docs_seen:
            docs_seen.add(doc_str)
            relevant_docs.append(doc_str)

        art_str = r.format_relevant_article()
        if art_str not in articles_seen and r.article_hint:
            articles_seen.add(art_str)
            relevant_articles.append(art_str)

    print(f"\n{hr('=')}")
    print(colorize("  [DINH DANG NOP BAI]", C.BOLD + C.YELLOW))
    print(hr('='))
    print(colorize("  relevant_docs:", C.CYAN))
    for d in relevant_docs:
        print(f"    - {d}")

    print(colorize("\n  relevant_articles:", C.CYAN))
    for a in relevant_articles:
        print(f"    - {a}")
    print(hr('='))


def print_document_summary(retriever: LegalRetriever, results: list[RetrievalResult]):
    """Tóm tắt kết quả theo văn bản pháp lý."""
    docs = retriever.aggregate_by_document(results)
    sorted_docs = sorted(docs.items(), key=lambda x: -x[1]["max_score"])

    print(f"\n{hr('-')}")
    print(colorize(f"  [TOM TAT THEO VAN BAN] ({len(docs)} van ban lien quan)", C.BOLD + C.CYAN))
    print(hr('-'))

    for doc_id, doc in sorted_docs:
        articles_str = ", ".join(doc["articles"]) if doc["articles"] else "-"
        print(f"  {colorize(doc['doc_number'], C.BOLD)}  {colorize(doc['legal_type'], C.GRAY)}")
        print(f"    Tieu de : {doc['title'][:70]}")
        print(f"    Chunks  : {len(doc['chunks'])}  |  Max score: {doc['max_score']:.4f}")
        print(f"    Dieu    : {colorize(articles_str, C.YELLOW)}")
        print()


# ─── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(retriever: LegalRetriever, query: str, top_k: int):
    print(f"\n{hr('=')}")
    print(colorize("  [BENCHMARK SO SANH 3 MODE]", C.BOLD + C.YELLOW))
    print(hr('='))

    for mode in ["fts", "vector", "hybrid"]:
        try:
            t0 = time.time()
            results = retriever.retrieve(query, mode=mode, top_k=top_k)
            elapsed = time.time() - t0
            mode_label = colorize(f"{mode.upper():<8}", C.CYAN)
            elapsed_label = colorize(f"{elapsed:.2f}s", C.GREEN if elapsed < 2 else C.YELLOW)
            print(f"  {mode_label} │ {len(results):2d} kết quả │ {elapsed_label}")
        except Exception as e:
            print(f"  {colorize(mode.upper(), C.RED):8s} │ LỖI: {e}")
    print(hr('='))


# ─── Main interactive loop ─────────────────────────────────────────────────────

def run_interactive(args):
    # Chọn retriever: local SQLite hoặc Supabase
    if getattr(args, "local", False):
        from retrieval.local_retriever import LocalRetriever
        retriever = LocalRetriever(
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            fts_weight=args.fts_weight,
            rrf_k=args.rrf_k,
        )
        backend = "LOCAL (SQLite)"
    else:
        # Thử kết nối Supabase, nếu lỗi fallback local
        try:
            retriever = LegalRetriever(
                top_k=args.top_k,
                vector_weight=args.vector_weight,
                fts_weight=args.fts_weight,
                rrf_k=args.rrf_k,
            )
            # Kiểm tra kết nối
            retriever._get_conn()
            backend = "SUPABASE (PostgreSQL)"
        except Exception as e:
            print(colorize(f"  [!] Supabase khong ket noi duoc: {e}", C.YELLOW))
            print(colorize("  [->] Tu dong chuyen sang LOCAL SQLite mode...", C.CYAN))
            from retrieval.local_retriever import LocalRetriever
            retriever = LocalRetriever(
                top_k=args.top_k,
                vector_weight=args.vector_weight,
                fts_weight=args.fts_weight,
                rrf_k=args.rrf_k,
            )
            backend = "LOCAL (SQLite - fallback)"

    # Khởi tạo Generator nếu chọn sinh câu trả lời
    generator = None
    if getattr(args, "llm", False):
        from retrieval.qwen_generator import QwenGenerator
        generator = QwenGenerator(model_name=getattr(args, "llm_model", "Qwen/Qwen3-8B-Instruct"))

    print_header()
    print(colorize(f"  Backend: {backend}", C.GREEN))
    print(colorize(f"  Mode: {args.mode.upper()}  |  Top-K: {args.top_k}  |  Rerank: {str(getattr(args, 'rerank', False)).upper()}  |  LLM: {str(getattr(args, 'llm', False)).upper()}", C.CYAN))
    if getattr(args, 'llm', False):
        print(colorize(f"  LLM Model: {args.llm_model}", C.YELLOW))
    print(colorize(f"  Vector weight: {args.vector_weight}  |  FTS weight: {args.fts_weight}  |  RRF-k: {args.rrf_k}", C.GRAY))
    print()
    print(colorize("  Nhap cau hoi phap ly. Go 'quit' de thoat.", C.WHITE))
    print(colorize("  Lenh: ':mode fts|vector|hybrid', ':gpt expand|hyde|none', ':rerank on|off', ':llm on|off', ':top 5', ':bench'", C.GRAY))
    print(hr())


    mode = args.mode
    top_k = args.top_k

    # Nếu có sẵn --query, chạy một lần rồi exit
    if args.query:
        queries = [args.query]
        interactive = False
    else:
        queries = []
        interactive = True

    while True:
        if interactive:
            try:
                raw = input(colorize("\n[?] Cau hoi: ", C.BOLD + C.GREEN)).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[+] Thoát.")
                break

            if not raw:
                continue
            if raw.lower() in ("quit", "exit", "q"):
                print("[+] Thoát.")
                break

            # Lệnh đặc biệt

            if raw.startswith(":mode "):
                new_mode = raw.split()[-1].lower()
                if new_mode in ("fts", "vector", "hybrid"):
                    mode = new_mode
                    print(colorize(f"  [OK] Da doi mode -> {mode.upper()}", C.GREEN))
                else:
                    print(colorize("  [ERR] Mode khong hop le (fts|vector|hybrid)", C.RED))
                continue

            if raw.startswith(":top "):
                try:
                    top_k = int(raw.split()[-1])
                    print(colorize(f"  [OK] Da doi top_k -> {top_k}", C.GREEN))
                except ValueError:
                    print(colorize("  [ERR] Gia tri khong hop le", C.RED))
                continue

            if raw.startswith(":rerank "):
                new_rerank = raw.split()[-1].lower()
                if new_rerank in ("on", "off", "true", "false"):
                    args.rerank = new_rerank in ("on", "true")
                    print(colorize(f"  [OK] Da doi rerank -> {str(args.rerank).upper()}", C.GREEN))
                else:
                    print(colorize("  [ERR] Rerank mode khong hop le (on|off)", C.RED))
                continue

            if raw.startswith(":llm "):
                new_llm = raw.split()[-1].lower()
                if new_llm in ("on", "off", "true", "false"):
                    args.llm = new_llm in ("on", "true")
                    if args.llm and generator is None:
                        from retrieval.qwen_generator import QwenGenerator
                        generator = QwenGenerator(model_name=getattr(args, "llm_model", "Qwen/Qwen3-8B-Instruct"))
                    print(colorize(f"  [OK] Da doi llm -> {str(args.llm).upper()}", C.GREEN))
                else:
                    print(colorize("  [ERR] LLM mode khong hop le (on|off)", C.RED))
                continue

            query = _sanitize_query(raw)
        else:
            if not queries:
                break
            query = _sanitize_query(queries.pop(0))

        # ── Benchmark ──
        if query == ":bench" or getattr(args, "benchmark", False):
            if not interactive:
                query = args.query
            else:
                try:
                    query = input(colorize("  Nhập câu hỏi để benchmark: ", C.YELLOW)).strip()
                except (EOFError, KeyboardInterrupt):
                    break
            run_benchmark(retriever, query, top_k)
            if not interactive:
                break
            continue

        # ── Tìm kiếm ──
        print(f"\n{hr()}")
        print(colorize(f"  [TIM KIEM] \"{query}\"  [mode={mode}, top_k={top_k}, rerank={getattr(args, 'rerank', False)}]", C.CYAN))
        print(hr())

        try:
            results = retriever.retrieve(query, mode=mode, top_k=top_k, rerank=getattr(args, "rerank", False))
        except Exception as e:
            print(colorize(f"  ✗ LỖI: {e}", C.RED))
            if not interactive:
                break
            continue

        if not results:
            print(colorize("  [!] Khong tim thay ket qua phu hop.", C.YELLOW))
            if not interactive:
                break
            continue

        # ── Hiển thị kết quả ──
        for i, r in enumerate(results, start=1):
            print_result(i, r, show_content=args.show_content)

        # ── Tóm tắt theo văn bản ──
        print_document_summary(retriever, results)

        # ── Định dạng nộp bài ──
        print_submission_format(results)

        # ── Sinh câu trả lời bằng LLM ──
        answer = ""
        if generator is not None:
            print(f"\n{hr()}")
            print(colorize("  [LLM] Đang sinh câu trả lời bằng local Qwen...", C.CYAN))
            print(hr())
            t0 = time.time()
            answer = generator.generate_answer(query, results[:5])
            print(colorize(f"  Thời gian sinh: {time.time() - t0:.2f}s", C.GRAY))
            print(colorize("\n  [CÂU TRẢ LỜI]:", C.BOLD + C.YELLOW))
            print(colorize(answer, C.WHITE))
            print(f"{hr()}")

        # ── Export JSON (nếu cần) ──
        if args.export or (interactive and input(
            colorize("\n  Export JSON? (y/N): ", C.GRAY)
        ).strip().lower() == "y"):
            export_path = args.export or "retrieval_output.json"
            export_data = {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "answer": answer,
                "results": [
                    {
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "chunk_index": r.chunk_index,
                        "doc_number": r.doc_number,
                        "title": r.title,
                        "legal_type": r.legal_type,
                        "score": r.score,
                        "source": r.source,
                        "article_hint": r.article_hint,
                        "content": r.content,
                        "relevant_doc": r.format_relevant_doc(),
                        "relevant_article": r.format_relevant_article(),
                    }
                    for r in results
                ],
            }
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(colorize(f"  [OK] Da export -> {export_path}", C.GREEN))

        if not interactive:
            break

    retriever.close()


# ─── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="R2AI Legal RAG — Retrieval Test (no LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Ví dụ:
          python retrieval/test_retrieval.py
          python retrieval/test_retrieval.py --mode fts --top-k 5
          python retrieval/test_retrieval.py --query "Doanh nghiệp nhỏ và vừa điều kiện thành lập"
          python retrieval/test_retrieval.py --mode hybrid --benchmark
          python retrieval/test_retrieval.py --query "Hợp đồng lao động" --export result.json
        """),
    )
    p.add_argument("--query", "-q", type=str, default=None,
                   help="Câu hỏi (nếu không truyền sẽ vào chế độ interactive)")
    p.add_argument("--mode", "-m", choices=["fts", "vector", "hybrid"],
                   default="hybrid", help="Chế độ tìm kiếm (mặc định: hybrid)")
    p.add_argument("--top-k", "-k", type=int, default=10,
                   help="Số kết quả trả về (mặc định: 10)")
    p.add_argument("--vector-weight", type=float, default=0.6,
                   help="Trọng số vector trong hybrid RRF (mặc định: 0.6)")
    p.add_argument("--fts-weight", type=float, default=0.4,
                   help="Trọng số FTS trong hybrid RRF (mặc định: 0.4)")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="Hằng số RRF k (mặc định: 60)")
    p.add_argument("--no-content", dest="show_content", action="store_false",
                   help="An noi dung chunk (chi hien metadata)")
    p.add_argument("--benchmark", "-b", action="store_true",
                   help="Chay benchmark so sanh ca 3 mode")
    p.add_argument("--export", type=str, default=None,
                   help="Export ket qua ra file JSON")

    p.add_argument("--rerank", "-r", action="store_true",
                   help="Sử dụng reranker model PhoRanker để tối ưu thứ tự kết quả")
    p.add_argument("--llm", action="store_true",
                   help="Kích hoạt mô hình sinh câu trả lời bằng LLM local")
    p.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Tên mô hình LLM trên HuggingFace (mặc định: Qwen/Qwen2.5-0.5B-Instruct)")
    p.add_argument("--local", "-l", action="store_true",
                   help="Dung local SQLite thay vi Supabase (offline mode)")
    return p


if __name__ == "__main__":
    # Enable ANSI colors on Windows terminal
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    args = build_parser().parse_args()
    run_interactive(args)
