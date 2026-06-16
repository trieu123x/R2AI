import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from retrieval.query_enhancer import GPTEnhancer

def main():
    print("=========================================================")
    print("   R2AI GPT-Enhanced Vector Retrieval Diagnostic Tool    ")
    print("=========================================================")
    
    print(f"[*] API Key status: {'CONFIGURED (Not empty)' if Config.OPENAI_API_KEY and Config.OPENAI_API_KEY != 'your_openai_api_key_here' else 'NOT CONFIGURED'}")
    
    enhancer = GPTEnhancer()
    print(f"[*] GPTEnhancer Active: {enhancer.is_active}")
    
    query = "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?"
    print(f"\n[*] Original Query:\n  \"{query}\"\n")
    
    print("[1] --- Query Expansion Mode ---")
    system_expand = (
        "Bạn là một trợ lý RAG chuyên về luật pháp Việt Nam.\n"
        "Hãy phân tích câu hỏi của người dùng và sinh ra một chuỗi từ khóa tìm kiếm mở rộng.\n"
        "Yêu cầu:\n"
        "1. Bao gồm các danh từ, thuật ngữ pháp luật chính thức đồng nghĩa (ví dụ: 'đấu thầu' -> 'lựa chọn nhà thầu', 'nhà thầu', 'gói thầu').\n"
        "2. Viết câu hỏi dưới dạng ngắn gọn, cô đọng.\n"
        "3. Trả về duy nhất chuỗi văn bản mở rộng chứa từ khóa và cụm từ tìm kiếm, KHÔNG giải thích gì thêm."
    )
    print("System Prompt:")
    print(f"  {system_expand.replace(chr(10), chr(10)+'  ')}")
    
    if enhancer.is_active:
        print("\nCalling OpenAI API for expansion...")
        expanded = enhancer.expand_query(query)
        print(f"Expanded Query: \"{expanded}\"")
    else:
        print("\n[!] OpenAI API key not configured, skipping live API call.")
        
    print("\n[2] --- HyDE (Hypothetical Document Embeddings) Mode ---")
    system_hyde = (
        "Bạn là một chuyên gia soạn thảo văn bản luật pháp Việt Nam.\n"
        "Hãy viết một đoạn điều khoản pháp lý giả định (ví dụ: một Điều trong Luật hoặc Nghị định) "
        "để trả lời trực tiếp cho câu hỏi của người dùng.\n"
        "Yêu cầu:\n"
        "1. Sử dụng văn phong pháp lý cực kỳ trang trọng, chính xác.\n"
        "2. Bắt đầu bằng 'Điều ...' và trình bày các khoản 1, 2, 3.\n"
        "3. Trả về duy nhất đoạn văn bản pháp lý giả định đó, KHÔNG chào hỏi hay giải thích gì thêm."
    )
    print("System Prompt:")
    print(f"  {system_hyde.replace(chr(10), chr(10)+'  ')}")
    
    if enhancer.is_active:
        print("\nCalling OpenAI API for HyDE document...")
        hyde = enhancer.generate_hyde(query)
        print(f"HyDE Document:\n{hyde}")
    else:
        print("\n[!] OpenAI API key not configured, skipping live API call.")
        
    print("\n=========================================================")
    print("   To enable live GPT features, please replace the value ")
    print("   of OPENAI_API_KEY in the .env file with your actual key.")
    print("=========================================================")

if __name__ == "__main__":
    main()
