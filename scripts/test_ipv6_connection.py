import socket
import psycopg2

ipv6_addr = "2406:da1a:82a:9d00:e46e:ddfb:a272:3e78"
port = 5432

print(f"Testing direct TCP connection to IPv6 address {ipv6_addr}:{port}...")
try:
    # Attempt to open socket
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect((ipv6_addr, port))
    print("[+] SUCCESS! Direct TCP connection established to the database via IPv6.")
    s.close()
    
    # Try connecting with psycopg2 using the IPv6 address
    print("\nTesting psycopg2 connection with IPv6 address...")
    # In PostgreSQL connection strings, IPv6 addresses must be enclosed in square brackets in the host parameter,
    # or passed as the host value directly in the connection string or connection params.
    conn_str = f"postgresql://postgres:Co7jrDKF7acp7V7O@[{ipv6_addr}]:5432/postgres"
    conn = psycopg2.connect(conn_str, connect_timeout=5)
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    print(f"[+] SUCCESS! psycopg2 connected. Server version: {cursor.fetchone()[0]}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"[-] Connection failed: {e}")
