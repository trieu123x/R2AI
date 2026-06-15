import urllib.request
import urllib.error

url = "https://sbbtftxbwkcatrecrvgk.supabase.co/rest/v1/"
service_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNiYnRmdHhid2tjYXRyZWNydmdrIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4MTQzMjc5NSwiZXhwIjoyMDk3MDA4Nzk1fQ.m1feuPFtyCYJcCdjYiAn8oC8QVkbIOkbmWam6i08efg"
headers = {
    "apikey": service_jwt,
    "Authorization": f"Bearer {service_jwt}",
    "User-Agent": "Mozilla/5.0"
}

print(f"Sending request to {url}...")
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        print("\nHeaders:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    print("\nHeaders (from HTTPError):")
    for k, v in e.headers.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
