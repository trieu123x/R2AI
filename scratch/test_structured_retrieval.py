import sqlite3
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def build_structured_fts_query(query: str) -> str:
    # Normalize query
    q_lower = query.lower()
    
    # Define semantic clusters and their synonyms
    # We want to identify if these clusters are active, and group them.
    # Subject clusters:
    subjects = {
        "incubator_coworking": {
            "keywords": ["ươm tạo", "làm việc chung"],
            "syns": ['"ươm tạo"', '"cơ sở ươm tạo"', '"khu làm việc chung"', '"làm việc chung"']
        },
        "sme": {
            "keywords": ["doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "sme"],
            "syns": ['"doanh nghiệp nhỏ và vừa"', '"nhỏ và vừa"', '"doanh nghiệp nhỏ"', '"doanh nghiệp siêu nhỏ"']
        },
        "labor": {
            "keywords": ["nhân viên", "lao động", "người lao động", "thử việc", "ký hợp đồng", "sa thải", "đuổi việc", "nghỉ việc", "lương"],
            "syns": ['"người lao động"', '"lao động"', '"nhân viên"', '"hợp đồng lao động"']
        }
    }
    
    # Topic / Action clusters:
    topics = {
        "tax_land": {
            "keywords": ["thuế", "đất đai", "mặt bằng", "tiền thuê"],
            "syns": ['"thuế"', '"đất đai"', '"mặt bằng"', '"thuê đất"', '"ưu đãi thuế"', '"tiền sử dụng đất"']
        },
        "bidding": {
            "keywords": ["đấu thầu", "nhà thầu", "gói thầu"],
            "syns": ['"đấu thầu"', '"nhà thầu"', '"lựa chọn nhà thầu"', '"gói thầu"', '"xếp hạng hồ sơ"']
        },
        "degrees_holding": {
            "keywords": ["giữ", "bằng", "văn bằng", "chứng chỉ", "giấy tờ", "bằng cấp"],
            "syns": ['"giữ"', '"thu giữ"', '"tạm giữ"', '"văn bằng"', '"chứng chỉ"', '"giấy tờ tùy thân"', '"bản chính"', '"bằng cấp"']
        }
    }

    # Find active subjects and topics
    active_subjects = []
    for s_name, s_info in subjects.items():
        if any(kw in q_lower for kw in s_info["keywords"]):
            active_subjects.append(s_info["syns"])
            
    active_topics = []
    for t_name, t_info in topics.items():
        if any(kw in q_lower for kw in t_info["keywords"]):
            active_topics.append(t_info["syns"])
            
    # If we found both subjects and topics, we AND them:
    # (subject_syn1 OR subject_syn2) AND (topic_syn1 OR topic_syn2)
    clauses = []
    if active_subjects:
        # Join subjects with OR
        flat_subjects = []
        for s in active_subjects:
            flat_subjects.extend(s)
        # remove duplicates
        flat_subjects = list(dict.fromkeys(flat_subjects))
        clauses.append("(" + " OR ".join(flat_subjects) + ")")
        
    if active_topics:
        flat_topics = []
        for t in active_topics:
            flat_topics.extend(t)
        flat_topics = list(dict.fromkeys(flat_topics))
        clauses.append("(" + " OR ".join(flat_topics) + ")")
        
    # If no structured clusters matched, fall back to extracting nouns/verbs
    if not clauses:
        # Simple token clean-up and OR
        words = [w.lower().strip() for w in query.split()]
        cleaned = []
        for w in words:
            w = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', w).strip()
            if w and len(w) >= 3:
                cleaned.append(f'"{w}"')
        if cleaned:
            return " OR ".join(cleaned)
        return ""
        
    return " AND ".join(clauses)

# Let's test the generator on the three questions
conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

questions = [
    "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
    "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
    "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
]

for idx, q in enumerate(questions, 1):
    fts_expr = build_structured_fts_query(q)
    print(f"\nQuestion {idx}: {q}")
    print(f"Generated FTS5 query: {fts_expr}")
    
    c.execute("""
        SELECT rowid, -bm25(chunks_fts5) AS score
        FROM chunks_fts5
        WHERE chunks_fts5 MATCH ?
        ORDER BY bm25(chunks_fts5)
        LIMIT 5
    """, (fts_expr,))
    rows = c.fetchall()
    print(f"Top 5 matches count: {len(rows)}")
    for r_idx, row in enumerate(rows, 1):
        rowid, score = row
        c.execute("SELECT id, content FROM document_chunks WHERE rowid = ?", (rowid,))
        cid, content = c.fetchone()
        # Parse document info
        print(f"  [{r_idx}] ID: {cid} | Score: {score:.4f}")
        print(f"      {content[:200]}")
