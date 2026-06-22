import json
import re
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Re-define our new matching logic
def clean_vietnamese_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[.,\-–/\"\'()“”:_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_doc_number(num: str) -> str:
    if not num:
        return ""
    num = num.lower().replace('đ', 'd').strip()
    return num

def is_doc_number_cited(doc_number: str, answer: str) -> bool:
    if not doc_number:
        return False
    ans_clean = clean_doc_number(answer)
    num_clean = clean_doc_number(doc_number)
    if num_clean in ans_clean:
        return True
    parts_slash = num_clean.split('/')
    if len(parts_slash) >= 2:
        base_slash = f"{parts_slash[0]}/{parts_slash[1]}"
        if base_slash in ans_clean:
            return True
    parts_dash = num_clean.split('-')
    if len(parts_dash) >= 2:
        base_dash = parts_dash[0]
        if '/' in base_dash and len(base_dash) >= 4:
            if base_dash in ans_clean:
                return True
    return False

def is_doc_name_mentioned(legal_type: str, title: str, answer: str) -> bool:
    if not title:
        return False
    legal_type = legal_type.strip() if legal_type else ""
    title = title.strip()
    candidates = []
    title_lower = title.lower()
    legal_lower = legal_type.lower()
    if legal_lower and title_lower.startswith(legal_lower):
        candidates.append(title)
    else:
        candidates.append(title)
        if legal_type:
            candidates.append(f"{legal_type} {title}")
    extended_candidates = []
    for c in candidates:
        extended_candidates.append(c)
        c_no_year = re.sub(r'\b(năm\s+)?20\d{2}\b', '', c, flags=re.IGNORECASE)
        c_no_year = re.sub(r'\b(năm\s+)?19\d{2}\b', '', c_no_year, flags=re.IGNORECASE)
        extended_candidates.append(c_no_year.strip())
    clean_ans = clean_vietnamese_text(answer)
    for c in extended_candidates:
        clean_c = clean_vietnamese_text(c)
        if len(clean_c.split()) >= 3:
            if clean_c in clean_ans:
                return True
    return False

def is_doc_cited(r_dict, answer: str) -> bool:
    doc_number = r_dict.get("doc_number", "")
    title = r_dict.get("title", "")
    legal_type = r_dict.get("legal_type", "")
    
    if is_doc_number_cited(doc_number, answer):
        return True
    if is_doc_name_mentioned(legal_type, title, answer):
        return True
    if legal_type and legal_type.lower() == "hiến pháp":
        if "hiến pháp" in answer.lower():
            return True
    return False

# Mock RetrievalResult format functions
def format_relevant_doc(r):
    return f"{r.get('doc_number')}|{r.get('legal_type')} {r.get('doc_number')} {r.get('title')}"

def format_relevant_article(r):
    doc_str = format_relevant_doc(r)
    article = r.get('article_hint') or "Toàn bộ"
    return f"{doc_str}|{article}"

# Load files
results = json.load(open('results.json', encoding='utf-8'))
cache_data = json.load(open('src/retrieval/stage1_cache.json', encoding='utf-8'))

# Index cache by qid
cache_map = {item['qid']: item for item in cache_data}

total_original_docs = 0
total_filtered_docs = 0
total_original_arts = 0
total_filtered_arts = 0

fallback_docs = 0
fallback_arts = 0
total_cases = 0

diff_examples = []

for entry in results:
    qid = entry['id']
    answer = entry['answer']
    orig_docs = entry['relevant_docs']
    orig_arts = entry['relevant_articles']
    
    total_original_docs += len(orig_docs)
    total_original_arts += len(orig_arts)
    
    if qid not in cache_map:
        # If not in cache, keep original
        total_filtered_docs += len(orig_docs)
        total_filtered_arts += len(orig_arts)
        continue
        
    total_cases += 1
    cache_entry = cache_map[qid]
    retrieved_results = cache_entry['results']
    
    # Filter
    docs_seen = set()
    articles_seen = set()
    filtered_docs = []
    filtered_articles = []
    
    for r in retrieved_results:
        doc_str = format_relevant_doc(r)
        art_str = format_relevant_article(r)
        
        if is_doc_cited(r, answer):
            if doc_str not in docs_seen:
                docs_seen.add(doc_str)
                filtered_docs.append(doc_str)
            
            # Check article
            article_hint = r.get('article_hint')
            if article_hint:
                m = re.search(r'\d+', article_hint)
                if m:
                    art_num = m.group()
                    art_pattern = rf'(?:Điều|khoản)\s+{art_num}\b'
                    if re.search(art_pattern, answer, re.IGNORECASE):
                        if art_str not in articles_seen:
                            articles_seen.add(art_str)
                            filtered_articles.append(art_str)
                else:
                    if art_str not in articles_seen:
                        articles_seen.add(art_str)
                        filtered_articles.append(art_str)
            else:
                if art_str not in articles_seen:
                    articles_seen.add(art_str)
                    filtered_articles.append(art_str)
                    
    # Fallback
    fb_doc = False
    fb_art = False
    
    if not filtered_docs and retrieved_results:
        fb_doc = True
        fallback_docs += 1
        top_r = retrieved_results[0]
        doc_str = format_relevant_doc(top_r)
        if doc_str not in docs_seen:
            docs_seen.add(doc_str)
            filtered_docs.append(doc_str)
            
    if not filtered_articles:
        fb_art = True
        for r in retrieved_results:
            doc_str = format_relevant_doc(r)
            if doc_str in docs_seen:
                art_str = format_relevant_article(r)
                if art_str not in articles_seen:
                    articles_seen.add(art_str)
                    filtered_articles.append(art_str)
                    
    if not filtered_articles and retrieved_results:
        fallback_arts += 1
        top_r = retrieved_results[0]
        art_str = format_relevant_article(top_r)
        if art_str not in articles_seen:
            articles_seen.add(art_str)
            filtered_articles.append(art_str)
            
    total_filtered_docs += len(filtered_docs)
    total_filtered_arts += len(filtered_articles)
    
    if len(orig_docs) != len(filtered_docs) or len(orig_arts) != len(filtered_articles):
        diff_examples.append({
            'id': qid,
            'orig_docs': orig_docs,
            'filtered_docs': filtered_docs,
            'answer': answer,
            'fallback_doc': fb_doc,
            'fallback_art': fb_art
        })

print(f"Total Cases Checked: {total_cases}")
print(f"Original Docs Count: {total_original_docs} | Filtered Docs Count: {total_filtered_docs}")
print(f"Original Arts Count: {total_original_arts} | Filtered Arts Count: {total_filtered_arts}")
print(f"Fallback Docs Triggered: {fallback_docs} ({fallback_docs/total_cases*100:.2f}%)")
print(f"Fallback Articles Triggered: {fallback_arts} ({fallback_arts/total_cases*100:.2f}%)")
print(f"Average Docs/Query - Original: {total_original_docs/len(results):.2f} | Filtered: {total_filtered_docs/len(results):.2f}")
print(f"Average Articles/Query - Original: {total_original_arts/len(results):.2f} | Filtered: {total_filtered_arts/len(results):.2f}")

print("\n=== TOP 5 EXAMPLES OF DOCUMENT COUNT CHANGE ===")
for ex in diff_examples[:5]:
    print("-" * 50)
    print(f"QID: {ex['id']} | Fallback: Doc={ex['fallback_doc']}, Art={ex['fallback_art']}")
    print(f"Original Docs: {ex['orig_docs']}")
    print(f"Filtered Docs: {ex['filtered_docs']}")
    print(f"Answer: {ex['answer'][:200]}...")
