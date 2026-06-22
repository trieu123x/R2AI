import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.batch_retrieve import build_submission_entry
from src.retrieval.retriever import RetrievalResult

def main():
    print("Running DB lookup test inside build_submission_entry...")
    
    # We will simulate a case where results list is empty or does not contain the cited article,
    # but the answer mentions "Điều 1" and "Nghị định 125/2020/NĐ-CP".
    qid = 1000
    question = "Test question"
    results = [] # Empty results to see if DB lookup fetches and adds the citation
    answer = "Theo quy định tại Điều 1 Nghị định 125/2020/NĐ-CP thì phạm vi điều chỉnh là..."
    
    entry = build_submission_entry(qid, question, results, answer)
    
    print("\nResulting Entry:")
    print(f"ID: {entry['id']}")
    print(f"Relevant Docs: {entry['relevant_docs']}")
    print(f"Relevant Articles: {entry['relevant_articles']}")
    
    # Check if they were successfully resolved
    if entry['relevant_docs'] and entry['relevant_articles']:
        print("\n[+] SUCCESS: DB lookup successfully recovered document and article from SQLite!")
    else:
        print("\n[-] FAILURE: DB lookup could not recover the citation.")

if __name__ == "__main__":
    main()
