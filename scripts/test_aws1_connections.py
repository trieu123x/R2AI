import psycopg2

project_ref = "sbbtftxbwkcatrecrvgk"
db_password = "Co7jrDKF7acp7V7O"
db_name = "postgres"

test_hosts = [
    "aws-1-ap-south-1.pooler.supabase.com",
    "aws-1-ap-southeast-1.pooler.supabase.com"
]

for host in test_hosts:
    for username in [f"postgres.{project_ref}", "postgres"]:
        print(f"\nTesting connection to {host} using user {username}...")
        conn_str = f"postgresql://{username}:{db_password}@{host}:5432/{db_name}"
        try:
            conn = psycopg2.connect(conn_str, connect_timeout=4)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"[+] SUCCESS! Connected to {host} using user {username}")
            print(f"    Server version: {version}")
            cursor.close()
            conn.close()
            # Stop if we succeed
            break
        except Exception as e:
            err_msg = str(e).strip().replace('\n', ' ')
            print(f"[-] Failed: {err_msg}")
