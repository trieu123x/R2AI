import psycopg2

project_ref = "sbbtftxbwkcatrecrvgk"
db_password = "Co7jrDKF7acp7V7O"
db_name = "postgres"

regions = [
    "ap-southeast-1", # Singapore
    "ap-south-1",     # Mumbai
    "ap-northeast-1", # Tokyo
    "ap-northeast-2", # Seoul
    "ap-southeast-2", # Sydney
    "us-east-1",      # N. Virginia
    "us-west-1",      # N. California
    "us-east-2",      # Ohio
    "us-west-2",      # Oregon
    "eu-west-1",      # Ireland
    "eu-central-1",   # Frankfurt
    "eu-west-2",      # London
    "eu-west-3",      # Paris
    "sa-east-1"       # Sao Paulo
]

for r in regions:
    host = f"aws-0-{r}.pooler.supabase.com"
    # Try username format: postgres.project_ref
    username = f"postgres.{project_ref}"
    conn_str = f"postgresql://{username}:{db_password}@{host}:5432/{db_name}"
    try:
        conn = psycopg2.connect(conn_str, connect_timeout=3)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"[+] SUCCESS! Connected to {host} using user {username}")
        print(f"    Server version: {version}")
        cursor.close()
        conn.close()
        break
    except Exception as e:
        err_msg = str(e).strip().replace('\n', ' ')
        print(f"[-] {host} (user {username}): {err_msg}")

    # Try username format: postgres
    username = "postgres"
    conn_str = f"postgresql://{username}:{db_password}@{host}:5432/{db_name}"
    try:
        conn = psycopg2.connect(conn_str, connect_timeout=3)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"[+] SUCCESS! Connected to {host} using user {username}")
        print(f"    Server version: {version}")
        cursor.close()
        conn.close()
        break
    except Exception as e:
        err_msg = str(e).strip().replace('\n', ' ')
        # We don't print connection timeouts for username 'postgres' if host is invalid
        if "timeout" not in err_msg:
            print(f"[-] {host} (user {username}): {err_msg}")
