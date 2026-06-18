import psycopg2

def main():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="Trieudh.1",
        database="law_vn"
    )
    cursor = conn.cursor()
    
    # Get active queries
    print("=== ACTIVE QUERIES ===")
    cursor.execute("""
        SELECT pid, query, state, wait_event_type, wait_event 
        FROM pg_stat_activity 
        WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%';
    """)
    for row in cursor.fetchall():
        print(row)
        
    # Get locks
    print("\n=== LOCKS ===")
    cursor.execute("""
        SELECT blocked_locks.pid     AS blocked_pid,
             blocked_activity.query  AS blocked_statement,
             blocking_locks.pid      AS blocking_pid,
             blocking_activity.query AS blocking_statement
        FROM  pg_catalog.pg_locks         blocked_locks
        JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
        JOIN pg_catalog.pg_locks         blocking_locks 
            ON blocking_locks.locktype = blocked_locks.locktype
            AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
            AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
            AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
            AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
            AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
            AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
            AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
            AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
            AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
            AND blocking_locks.pid != blocked_locks.pid
        JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
        WHERE NOT blocked_locks.granted;
    """)
    for row in cursor.fetchall():
        print(row)
        
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
