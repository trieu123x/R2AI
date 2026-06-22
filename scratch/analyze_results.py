import json
import os
import sys

sys.stdout = io = open(sys.stdout.fileno(), mode='w', encoding='utf8', closefd=False)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results.json")

if not os.path.exists(RESULTS_PATH):
    print(f"File not found: {RESULTS_PATH}")
    sys.exit(1)

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

doc_counts = []
art_counts = []
empty_answers = 0
fallback_answers = 0

for item in data:
    docs = item.get("relevant_docs", [])
    arts = item.get("relevant_articles", [])
    doc_counts.append(len(docs))
    art_counts.append(len(arts))
    ans = item.get("answer", "")
    if not ans:
        empty_answers += 1
    elif "Trả lời trực tiếp:" in ans or "Hạn chế của dữ liệu" in ans:
        fallback_answers += 1

print(f"Total questions: {len(data)}")
print(f"Empty answers: {empty_answers}")
print(f"Fallback/Rule-based answers: {fallback_answers}")
print(f"Average relevant_docs per question: {sum(doc_counts)/len(data):.2f} (Min: {min(doc_counts)}, Max: {max(doc_counts)})")
print(f"Average relevant_articles per question: {sum(art_counts)/len(data):.2f} (Min: {min(art_counts)}, Max: {max(art_counts)})")

# Print distribution of doc counts
from collections import Counter
print("\nDoc count distribution:")
for k, v in sorted(Counter(doc_counts).items()):
    print(f"  {k} docs: {v} questions")

print("\nArticle count distribution:")
for k, v in sorted(Counter(art_counts).items()):
    print(f"  {k} articles: {v} questions")
