import socket
import urllib.request

print("Testing DNS resolution for google.com...")
try:
    ip = socket.gethostbyname("google.com")
    print(f"[+] google.com resolved to {ip}")
except Exception as e:
    print(f"[-] Failed to resolve google.com: {e}")

print("\nTesting HTTP request to google.com...")
try:
    response = urllib.request.urlopen("https://www.google.com", timeout=5)
    print(f"[+] HTTP request succeeded. Status code: {response.status}")
except Exception as e:
    print(f"[-] HTTP request failed: {e}")
