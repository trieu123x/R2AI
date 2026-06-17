import os
import sys
import time
import torch
import io

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

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    retriever = LegalRetriever(use_postgres=False, top_k=5)
    query = "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?"
    results = retriever.retrieve(query, mode="hybrid", top_k=5)
    
    generator = QwenGenerator(model_name=model_name)
    generator._lazy_load()
    
    settings = [
        {"name": "rep=1.05, no_repeat=0", "rep": 1.05},
        {"name": "rep=1.08, no_repeat=0", "rep": 1.08},
        {"name": "rep=1.10, no_repeat=0", "rep": 1.10}
    ]
    
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
        "- Tuyệt đối KHÔNG sử dụng tên văn bản, điều luật hoặc các từ ngữ trong Ví dụ mẫu (Few-shot) như 'Nghị định X', 'quyền công đoàn', 'thành lập công đoàn' để trả lời cho câu hỏi thực tế.\n"
        "- Cực kỳ cẩn thận với các từ phủ định hoặc hành vi cấm: các cụm từ như 'nghiêm cấm', 'không được', 'xử phạt đối với hành vi' có nghĩa là hành vi đó BỊ CẤM, tuyệt đối không được viết thành 'người sử dụng lao động được phép thực hiện'.\n"
        "- Không chỉ nêu tên văn bản chung chung. Hãy giải thích cụ thể nội dung quyền lợi, nghĩa vụ, ưu đãi hoặc mức phạt được quy định.\n\n"
        "Bạn BẮT BUỘC phải trình bày câu trả lời của mình nghiêm ngặt theo cấu trúc 4 phần sau (sử dụng chính xác các tiêu đề này làm tiêu đề dòng):\n"
        "1. Trả lời trực tiếp: Trả lời trực tiếp vào câu hỏi, nêu rõ kết luận chính hoặc hành vi và hệ quả.\n"
        "2. Phân tích chi tiết: Diễn giải chi tiết nội dung quy định, mức phạt bằng tiền cụ thể, quyền lợi/nghĩa vụ chi tiết được quy định trong các tài liệu tham khảo.\n"
        "3. Căn cứ pháp lý: Liệt kê chi tiết các điều khoản, điều luật cụ thể được sử dụng làm căn cứ từ tài liệu tham khảo (Ví dụ: 'Điều 17 Bộ luật Lao động 2019', 'Điều 15 Nghị định 12/2022/NĐ-CP').\n"
        "4. Hạn chế của dữ liệu (nếu có): Nêu rõ nếu các căn cứ pháp lý được cung cấp thiếu thông tin hoặc không đủ cơ sở để trả lời đầy đủ một khía cạnh nào đó của câu hỏi."
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
        {"role": "user", "content": (
            "TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
            "=========================================\n"
            "[Căn cứ 1]\n"
            "Độ liên quan: 0.9876\n"
            "Văn bản: Nghị định X năm 2022 về xử phạt hành chính\n"
            "Điều: Điều 20\n"
            "Nội dung:\n"
            "Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập công đoàn.\n"
            "=========================================\n\n"
            "CÂU HỎI: Hành vi cản trở người lao động thành lập công đoàn bị phạt bao nhiêu tiền?\n\n"
            "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên."
        )},
        {"role": "assistant", "content": (
            "1. Trả lời trực tiếp: Hành vi cản trở người lao động thành lập công đoàn sẽ bị xử phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng.\n\n"
            "2. Phân tích chi tiết:\n"
            "- Về hành vi vi phạm: Người sử dụng lao động có hành vi cản trở người lao động thành lập công đoàn bị nghiêm cấm theo quy định.\n"
            "- Về mức phạt: Hành vi vi phạm này sẽ bị xử phạt hành chính với mức phạt tiền cụ thể từ 10.000.000 đồng đến 20.000.000 đồng.\n\n"
            "3. Căn cứ pháp lý:\n"
            "- Khoản 1 Điều 20 Nghị định X năm 2022.\n\n"
            "4. Hạn chế của dữ liệu (nếu có):\n"
            "- Tài liệu được cung cấp không đề cập đến các hình thức xử phạt bổ sung hay biện pháp khắc phục hậu quả khác đối với hành vi này."
        )},
        {"role": "user", "content": user_content}
    ]
    
    prompt = generator._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = generator._tokenizer([prompt], return_tensors="pt").to(generator._model.device)
    
    for s in settings:
        print(f"\nEvaluating: {s['name']}")
        t0 = time.time()
        with torch.no_grad():
            outputs = generator._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=s["rep"],
                pad_token_id=generator._tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        gen_ids = outputs[0][len(inputs.input_ids[0]):]
        response = generator._tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"Generated {len(gen_ids)} tokens in {elapsed:.2f}s")
        print("ANSWER OUTPUT:")
        print(response)
        print("="*60)

if __name__ == "__main__":
    main()
