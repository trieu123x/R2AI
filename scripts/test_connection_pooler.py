import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from config import Config

# Parse project ref from original connection string:
# postgresql://postgres:[YOUR-PASSWORD]@db.sbbtftxbwkcatrecrvgk.supabase.co:5432/postgres
# The project ref is "sbbtftxbwkcatrecrvgk"
project_ref = "sbbtftxbwkcatrecrvgk"
db_user = f"postgres.{project_ref}"
db_password = "Co7jrDKF7acp7V7O"
db_name = "postgres"

test_hosts = [
    "aws-0-ap-south-1.pooler.supabase.com",      # Mumbai (matches the IPv6 range)
    "aws-0-ap-southeast-1.pooler.supabase.com"   # Singapore (typical default for VN)
]

for host in test_hosts:
    print(f"\nTesting connection to {host} on port 5432 (Session mode)...")
    # For connection pooler, port 5432 is Session mode, 6543 is Transaction mode.
    # Standard psycopg2 can connect to either. Let's try 5432 first.
    conn_str = f"postgresql://{db_user}:{db_password}@{host}:5432/{db_name}"
    try:
        conn = psycopg2.connect(conn_str, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"[+] SUCCESS! Connected to {host}")
        print(f"    Server version: {version}")
        cursor.close()
        conn.close()
        # If we succeed, let's update .env with this connection string!
        break
    except Exception as e:
        print(f"[-] Failed to connect to {host}: {e}")
