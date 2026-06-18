import os
import sys
import time
import torch
import io
import re

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.qwen_generator import QwenGenerator, extract_evidence
from retrieval.retriever import LegalRetriever

def has_repetitive_loop(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for i in range(len(lines) - 2):
        if lines[i] == lines[i+1] == lines[i+2]:
            return True
            
    for line in lines:
        words = line.split()
        if len(words) > 15:
            for size in range(2, 9):
                for start in range(len(words) - 3 * size):
                    w1 = words[start : start + size]
                    w2 = words[start + size : start + 2 * size]
                    w3 = words[start + 2 * size : start + 3 * size]
                    if w1 == w2 == w3:
                        return True
    return False

def check_structure_and_quality(answer: str) -> tuple:
    # Check if 4 parts exist in the exact titles or close enough
    headers = [
        "1. Trả lời trực tiếp",
        "2. Phân tích chi tiết",
        "3. Căn cứ pháp lý",
        "4. Hạn chế của dữ liệu"
    ]
    missing = []
    for h in headers:
        if h.lower() not in answer.lower():
            missing.append(h)
    
    # Check for Chinese/English mixed characters that suggest degeneration
    # e.g., Chinese characters 
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', answer)
    has_chinese = len(chinese_chars) > 0
    
    # Check for typical degraded spellings or wordings
    degraded_words = ["chánh", "work chánh", "works chánh", "chiết khấu", "trực tuyến", "kilobyus"]
    found_degraded = [w for w in degraded_words if w in answer.lower()]
    
    return missing, has_chinese, found_degraded

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Initializing retriever...")
    retriever = LegalRetriever(use_postgres=False, top_k=5)
    
    query = "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?"
    results = retriever.retrieve(query, mode="hybrid", top_k=5)
    
    print("Loading generator...")
    generator = QwenGenerator(model_name=model_name)
    generator._lazy_load()
    
    # Let's prepare context
    context_parts = []
    for idx, r in enumerate(results[:3], start=1):
        extracted_body = extract_evidence(r.content, r.article_hint)
        lines = extracted_body.split('\n')
        body_lines = []
        for line in lines:
            if not line.strip().startswith("Văn bản:"):
                body_lines.append(line)
        clean_body = "\n".join(body_lines).strip()
        part = (
            f"[Căn cứ {idx}]\n"
            f"Văn bản: {r.legal_type} số {r.doc_number} {r.title}\n"
            f"Điều: {r.article_hint or 'Toàn bộ'}\n"
            f"Nội dung:\n{clean_body}"
        )
        context_parts.append(part)
    context = "\n\n=========================================\n\n".join(context_parts)
    
    system_prompt = (
        "Bạn là một trợ lý pháp lý Việt Nam chuyên nghiệp, chính xác và đáng tin cậy.\n\n"
        "Nhiệm vụ của bạn:\n"
        "- Trả lời câu hỏi dựa trên CÁC CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP. Tuyệt đối không tự suy diễn hoặc giả định các nội dung không có trong tài liệu.\n"
        "- Hãy viết hoàn toàn bằng tiếng Việt chuẩn mực. Tuyệt đối KHÔNG trộn lẫn tiếng Trung (chữ Hán), tiếng Anh hoặc bất kỳ từ ngữ nước ngoài nào khác.\n"
        "- Tuyệt đối KHÔNG được tự bịa ra các số hiệu văn bản, điều luật không có trong các căn cứ pháp lý.\n"
        "- Không chỉ nêu tên văn bản chung chung. Hãy giải thích cụ thể nội dung quyền lợi, nghĩa vụ, ưu đãi hoặc mức phạt được quy định.\n\n"
        "Bạn BẮT BUỘC phải trình bày câu trả lời của mình nghiêm ngặt theo cấu trúc 4 phần sau:\n"
        "1. Trả lời trực tiếp: Trả lời trực tiếp vào câu hỏi, nêu rõ kết luận chính.\n"
        "2. Phân tích chi tiết: Diễn giải chi tiết nội dung quy định, ưu đãi hoặc mức phạt từ tài liệu tham khảo.\n"
        "3. Căn cứ pháp lý: Liệt kê chi tiết các điều khoản cụ thể sử dụng làm căn cứ.\n"
        "4. Hạn chế của dữ liệu (nếu có): Nêu rõ nếu các căn cứ pháp lý thiếu thông tin."
    )
    
    user_content = (
        f"TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
        f"=========================================\n"
        f"{context}\n"
        f"=========================================\n\n"
        f"CÂU HỎI: {query}\n\n"
        f"Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên.\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        # Few-shot example
        {"role": "user", "content": (
            "TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
            "=========================================\n"
            "[Căn cứ 1]\n"
            "Văn bản: Nghị định 39/2018/NĐ-CP\n"
            "Điều: Điều 15\n"
            "Nội dung:\n"
            "Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở doanh nghiệp nhỏ và vừa.\n"
            "=========================================\n\n"
            "CÂU HỎI: Hành vi cản trở doanh nghiệp nhỏ và vừa bị phạt bao nhiêu tiền?\n\n"
            "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên."
        )},
        {"role": "assistant", "content": (
            "1. Trả lời trực tiếp: Hành vi cản trở doanh nghiệp nhỏ và vừa sẽ bị xử phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng.\n\n"
            "2. Phân tích chi tiết:\n"
            "- Về hành vi vi phạm: Các hành vi cản trở hoạt động của doanh nghiệp nhỏ và vừa bị nghiêm cấm.\n"
            "- Về mức phạt: Mức phạt cụ thể từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở này.\n\n"
            "3. Căn cứ pháp lý:\n"
            "- Điều 15 Nghị định 39/2018/NĐ-CP.\n\n"
            "4. Hạn chế của dữ liệu (nếu có):\n"
            "- Tài liệu được cung cấp không đề cập đến các hình thức xử phạt bổ sung khác."
        )},
        {"role": "user", "content": user_content}
    ]
    
    prompt = generator._tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = generator._tokenizer([prompt], return_tensors="pt").to(generator._model.device)
    
    # Try different generation configurations
    configs = [
        # 1. Greedy default
        {"name": "Greedy, rep=1.05, no_repeat=0", "do_sample": False, "rep": 1.05, "no_repeat": 0},
        {"name": "Greedy, rep=1.1, no_repeat=0", "do_sample": False, "rep": 1.1, "no_repeat": 0},
        {"name": "Greedy, rep=1.1, no_repeat=8", "do_sample": False, "rep": 1.1, "no_repeat": 8},
        {"name": "Greedy, rep=1.05, no_repeat=12", "do_sample": False, "rep": 1.05, "no_repeat": 12},
        {"name": "Greedy, rep=1.08, no_repeat=15", "do_sample": False, "rep": 1.08, "no_repeat": 15},
        # 2. Sampling
        {"name": "Sample T=0.2, rep=1.05, no_repeat=0", "do_sample": True, "temperature": 0.2, "top_p": 0.9, "rep": 1.05, "no_repeat": 0},
        {"name": "Sample T=0.3, rep=1.05, no_repeat=0", "do_sample": True, "temperature": 0.3, "top_p": 0.9, "rep": 1.05, "no_repeat": 0},
        {"name": "Sample T=0.5, rep=1.05, no_repeat=0", "do_sample": True, "temperature": 0.5, "top_p": 0.9, "rep": 1.05, "no_repeat": 0},
        {"name": "Sample T=0.3, rep=1.05, no_repeat=12", "do_sample": True, "temperature": 0.3, "top_p": 0.9, "rep": 1.05, "no_repeat": 12},
        {"name": "Sample T=0.3, rep=1.08, no_repeat=15", "do_sample": True, "temperature": 0.3, "top_p": 0.9, "rep": 1.08, "no_repeat": 15},
    ]
    
    for cfg in configs:
        print("\n" + "="*80)
        print(f"TESTING CONFIG: {cfg['name']}")
        print("="*80)
        
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": cfg["do_sample"],
            "repetition_penalty": cfg["rep"],
            "pad_token_id": generator._tokenizer.eos_token_id,
        }
        if cfg["do_sample"]:
            gen_kwargs["temperature"] = cfg["temperature"]
            gen_kwargs["top_p"] = cfg["top_p"]
        if cfg["no_repeat"] > 0:
            gen_kwargs["no_repeat_ngram_size"] = cfg["no_repeat"]
            
        t0 = time.time()
        with torch.no_grad():
            outputs = generator._model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        response = generator._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        num_tokens = len(generated_ids[0])
        tok_speed = num_tokens / elapsed if elapsed > 0 else 0
        
        is_loop = has_repetitive_loop(response)
        missing_parts, has_chinese, degraded = check_structure_and_quality(response)
        
        print(f"Stats: {num_tokens} tokens in {elapsed:.2f}s ({tok_speed:.2f} tok/s)")
        print(f"Validation:")
        print(f"  - Has Repetitive Loop: {is_loop}")
        print(f"  - Missing Parts: {missing_parts}")
        print(f"  - Has Chinese: {has_chinese}")
        print(f"  - Degraded Words: {degraded}")
        print("-" * 40)
        print("OUTPUT:")
        print(response)
        print("="*80)

if __name__ == "__main__":
    main()
