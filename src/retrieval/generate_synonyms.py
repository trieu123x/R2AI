import os
import sys
import json
import urllib.request
import urllib.error
import io
import re

# Cấu hình UTF-8 cho Windows để in được tiếng Việt không bị lỗi font/mã hóa
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".model_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

# Thêm PROJECT_ROOT vào path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config

# Các thực thể luật pháp Việt Nam cốt lõi cần sinh từ đồng nghĩa
CORE_LEGAL_TERMS = [
    "sme",
    "sa thải",
    "thai sản",
    "bảo hiểm xã hội",
    "giấy tờ",
    "công đoàn",
    "thuế thu nhập doanh nghiệp",
    "hợp đồng lao động",
    "đăng ký kinh doanh",
    "thuế giá trị gia tăng"
]

DEFAULT_DICT = {
    "sme": ["doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "doanh nghiệp siêu nhỏ"],
    "sa_thải": ["đuổi việc", "chấm dứt hợp đồng lao động", "kỷ luật sa thải"],
    "thai_sản": ["nghỉ đẻ", "mang thai", "sinh con"],
    "bảo_hiểm_xã_hội": ["bhxh", "bảo hiểm", "đóng bảo hiểm"],
    "giấy_tờ": ["bản chính", "bằng cấp", "chứng chỉ", "văn bằng", "giấy tờ tùy thân"],
    "công_đoàn": ["tổ chức công đoàn", "đại diện người lao động"],
    "thuế_thu_nhập_doanh_nghiệp": ["thuế tndn", "thuế doanh nghiệp", "nộp thuế tndn"],
    "hợp_đồng_lao_động": ["hđlđ", "hợp đồng làm việc", "ký hợp đồng"],
    "đăng_ký_kinh_doanh": ["đăng ký doanh nghiệp", "giấy phép kinh doanh", "thành lập doanh nghiệp"],
    "thuế_giá_trị_gia_tăng": ["thuế gtgt", "thuế vat", "vat"]
}

def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so sánh chuỗi chính xác."""
    patterns = {
        '[àáảãạăằắẳẵặâầấẩẫậ]': 'a',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ìíỉĩị]': 'i',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ùúủũụưừứửữự]': 'u',
        '[ỳýỷỹỵ]': 'y',
        '[đ]': 'd',
        '[ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]': 'A',
        '[ÈÉẺẼẸÊỀẾỂỄỆ]': 'E',
        '[ÌÍỈĨỊ]': 'I',
        '[ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ]': 'O',
        '[ÙÚỦŨỤƯỪỨỬỮỰ]': 'U',
        '[ỲÝỶỸỴ]': 'Y',
        '[Đ]': 'D'
    }
    for pattern, replace in patterns.items():
        text = re.sub(pattern, replace, text)
    return text

def is_valid_synonym(synonym: str, current_key: str, core_terms: list) -> bool:
    """
    Kiểm tra tính hợp lệ của từ đồng nghĩa để tránh ô nhiễm dữ liệu.
    Loại bỏ các từ trùng với chính từ khóa gốc, hoặc trùng/chứa các từ khóa gốc khác.
    """
    syn_clean = synonym.lower().strip()
    key_clean = current_key.lower().replace("_", " ").strip()
    
    # 1. Không trùng với chính từ khóa gốc
    if syn_clean == key_clean or syn_clean.replace(" ", "_") == current_key:
        return False
        
    # 2. Không được rỗng hoặc quá ngắn/quá dài hoặc chứa ký tự đặc biệt
    if len(syn_clean) <= 2 or len(syn_clean) > 50 or any(c in syn_clean for c in ["{", "}", "[", "]", ":", "\n"]):
        return False
        
    # 3. Không được trùng hoặc chứa bất kỳ từ khóa gốc khác (tránh cross-pollution)
    syn_no_accent = strip_accents(syn_clean)
    for term in core_terms:
        term_clean = term.lower().strip()
        if term_clean == key_clean:
            continue
        term_no_accent = strip_accents(term_clean)
        
        # Nếu từ đồng nghĩa chứa từ khóa khác hoặc ngược lại (ví dụ: "thái sản" chứa "thai sản")
        if term_no_accent in syn_no_accent or syn_no_accent in term_no_accent:
            return False
            
    return True

def generate_via_openai(api_key: str, prompt: str) -> str:
    """Gọi OpenAI API để sinh từ điển đồng nghĩa."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Bạn là chuyên gia luật Việt Nam giúp xây dựng từ điển đồng nghĩa."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[-] Lỗi khi gọi OpenAI API: {e}")
        return ""

def generate_via_local_llm(prompt: str) -> str:
    """Tải model Qwen nhỏ chạy offline để sinh từ điển đồng nghĩa, lưu cache tại .model_cache."""
    print("[+] Đang load mô hình Qwen/Qwen2.5-0.5B-Instruct cục bộ (lưu cache tại .model_cache)...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    messages = [
        {"role": "system", "content": "Bạn là chuyên gia luật Việt Nam giúp xây dựng từ điển đồng nghĩa dạng JSON."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.01,
        do_sample=False
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response.strip()

def main():
    print("[+] Bắt đầu quá trình sinh từ điển đồng nghĩa pháp lý bằng AI...")
    
    # Tạo danh sách các khóa mong muốn dưới dạng gạch dưới để LLM điền vào
    expected_keys = [t.lower().replace(" ", "_") for t in CORE_LEGAL_TERMS]
    keys_str = "\n".join(f"   - \"{k}\"" for k in expected_keys)
    
    # Chuẩn bị Prompt yêu cầu xuất JSON cấu trúc chuẩn, tránh cung cấp ví dụ có giá trị cụ thể để tránh model copy-paste ví dụ
    prompt = (
        "Nhiệm vụ: Hãy tạo một từ điển từ đồng nghĩa tiếng Việt cho các cụm từ pháp lý sau đây:\n"
        f"Danh sách từ khóa gốc: {', '.join(CORE_LEGAL_TERMS)}\n\n"
        "Yêu cầu:\n"
        "1. Trả về kết quả duy nhất là 1 đối tượng JSON.\n"
        "2. Trong đối tượng JSON, bắt buộc phải sinh đầy đủ các keys tương ứng với các từ khóa gốc viết thường, khoảng trắng được thay thế bằng dấu gạch dưới '_' như sau:\n"
        f"{keys_str}\n"
        "3. Không được bỏ sót bất kỳ từ khóa nào ở trên.\n"
        "4. Value của mỗi key là một danh sách (Array) chứa các từ hoặc cụm từ đồng nghĩa phổ biến trong văn phong pháp lý hoặc đời sống tại Việt Nam (từ 3 - 5 từ mỗi nhóm).\n"
        "5. Không thêm giải thích, không bọc markdown ```json ... ```, chỉ trả về chuỗi JSON thô.\n\n"
        "Mẫu cấu trúc JSON yêu cầu (hãy điền các từ đồng nghĩa thực tế thay thế cho từ_đồng_nghĩa_A, từ_đồng_nghĩa_B,...):\n"
        "{\n"
        "  \"từ_khóa_gốc_1\": [\"từ_đồng_nghĩa_A\", \"từ_đồng_nghĩa_B\", \"từ_đồng_nghĩa_C\"],\n"
        "  \"từ_khóa_gốc_2\": [\"từ_đồng_nghĩa_D\", \"từ_đồng_nghĩa_E\", \"từ_đồng_nghĩa_F\"]\n"
        "}"
    )

    api_key = Config.OPENAI_API_KEY
    raw_json = ""
    is_openai = False
    
    # 1. Thử dùng OpenAI nếu có key hợp lệ
    if api_key and api_key != "your_openai_api_key_here":
        print("[+] Phát hiện OpenAI API Key. Đang sinh từ điển qua OpenAI...")
        raw_json = generate_via_openai(api_key, prompt)
        if raw_json:
            is_openai = True
        
    # 2. Nếu OpenAI thất bại hoặc không có key, dùng local HuggingFace Qwen
    if not raw_json:
        print("[!] Không có OpenAI API Key hoặc gọi API thất bại. Đang chuyển sang sử dụng mô hình Qwen cục bộ...")
        print("[!] Cảnh báo: Mô hình cục bộ Qwen 0.5B có dung lượng nhỏ nên độ chính xác từ đồng nghĩa tiếng Việt sẽ hạn chế hơn so với OpenAI GPT.")
        try:
            raw_json = generate_via_local_llm(prompt)
        except Exception as e:
            print(f"[-] Không thể chạy mô hình cục bộ: {e}")
            
    # 3. Phân tích kết quả JSON
    synonyms_dict = None
    if raw_json:
        try:
            # Làm sạch nếu model bọc markdown ```json
            if "```" in raw_json:
                # Tìm đoạn JSON nằm giữa các dấu ```
                for block in raw_json.split("```"):
                    block = block.strip()
                    if block.startswith("json"):
                        block = block[4:].strip()
                    if block.startswith("{") and block.endswith("}"):
                        try:
                            synonyms_dict = json.loads(block)
                            break
                        except:
                            pass
            
            if not synonyms_dict:
                # Thử load trực tiếp từ chuỗi thô
                cleaned_raw = raw_json.strip()
                if cleaned_raw.startswith("```"):
                    cleaned_raw = cleaned_raw.replace("```", "")
                if cleaned_raw.startswith("json"):
                    cleaned_raw = cleaned_raw[4:].strip()
                synonyms_dict = json.loads(cleaned_raw)
                
            print("[+] Phân tích chuỗi JSON từ AI thành công!")
        except Exception as e:
            print(f"[-] Lỗi phân tích JSON trả về từ AI: {e}\nNội dung thô: {raw_json}")

    # 4. Xử lý và Hợp nhất dữ liệu
    final_dict = DEFAULT_DICT.copy()
    
    if synonyms_dict:
        print("[+] Tiến hành chuẩn hóa và lọc từ đồng nghĩa từ AI...")
        for k, v in synonyms_dict.items():
            if isinstance(v, list) and v:
                k_clean = k.lower().strip().replace(" ", "_")
                # Lọc và chuẩn hóa danh sách các từ đồng nghĩa từ AI
                v_clean = []
                for item in v:
                    if isinstance(item, str) and is_valid_synonym(item, k_clean, CORE_LEGAL_TERMS):
                        v_clean.append(item.strip().lower())
                
                if not v_clean:
                    continue
                
                if is_openai:
                    # Ghi đè bằng kết quả chất lượng cao của OpenAI
                    final_dict[k_clean] = v_clean
                else:
                    # Với local LLM, chỉ thêm các từ mới hợp lệ vào danh sách mặc định (tránh trùng lặp)
                    existing = final_dict.get(k_clean, [])
                    new_terms = [t for t in v_clean if t not in existing]
                    if new_terms:
                        final_dict[k_clean] = existing + new_terms
    else:
        print("[!] Không có kết quả từ AI hoặc phân tích thất bại. Sử dụng từ điển mặc định...")

    # Ghi file synonyms.json
    output_path = os.path.join(PROJECT_ROOT, "src", "retrieval", "synonyms.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=2)
        print(f"[+] Đã ghi thành công từ điển đồng nghĩa pháp lý tại: {output_path}")
    except Exception as e:
        print(f"[-] Lỗi khi ghi file synonyms.json: {e}")

if __name__ == "__main__":
    main()
