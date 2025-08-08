import psycopg2

try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='fraud_detection',
        user='fraud_admin',
        password='FraudDetection2024!'
    )
    cur = conn.cursor()
    
    # Get all tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row[0] for row in cur.fetchall()]
    print("Available tables:")
    for table in tables:
        print(f"  - {table}")
    
    # Check transactions table structure
    if 'transactions' in tables:
        print("\nTransactions table columns:")
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'transactions'")
        columns = cur.fetchall()
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")