import urllib.request
import json

types = ["CNAME", "AAAA", "A"]
for t in types:
    url = f"https://dns.google/resolve?name=db.sbbtftxbwkcatrecrvgk.supabase.co&type={t}"
    print(f"\nQuerying Google DoH for type {t}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            if "Answer" in data:
                for ans in data["Answer"]:
                    print(f"  {ans['name']} -> type {ans['type']} -> {ans['data']}")
            else:
                print("  No answer section.")
    except Exception as e:
        print(f"  Error: {e}")
