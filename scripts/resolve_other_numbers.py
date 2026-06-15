import socket

numbers = [0, 1, 2, 3, 4, 5]
regions = ["ap-southeast-1", "ap-south-1"]

print("Testing other pooler index numbers...")
for n in numbers:
    for r in regions:
        host = f"aws-{n}-{r}.pooler.supabase.com"
        try:
            ip = socket.gethostbyname(host)
            print(f"[+] {host} resolved to {ip}")
        except Exception as e:
            pass
