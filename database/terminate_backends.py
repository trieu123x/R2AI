import psycopg2

def main():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="Trieudh.1",
        database="law_vn"
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE datname = 'law_vn' AND pid != pg_backend_pid();
    """)
    terminated = cursor.fetchall()
    print(f"Terminated {len(terminated)} backends.")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
