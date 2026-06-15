import socket

regions = [
    "ap-southeast-1", # Singapore
    "ap-south-1",     # Mumbai
    "ap-northeast-1", # Tokyo
    "ap-northeast-2", # Seoul
    "us-east-1",     # N. Virginia
    "us-west-1",     # N. California
    "eu-west-1",      # Ireland
    "eu-central-1"    # Frankfurt
]

print("Testing Supabase pooler hostnames...")
for r in regions:
    host = f"aws-0-{r}.pooler.supabase.com"
    try:
        ip = socket.gethostbyname(host)
        print(f"[+] {host} resolved to {ip}")
    except Exception as e:
        # Try without aws-0- (e.g. just pooler.supabase.com?)
        # Or maybe some regions use different numbering
        pass
