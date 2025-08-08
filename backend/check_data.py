import psycopg2
import os

# Database connection
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='fraud_detection',
    user='fraud_admin',
    password='FraudDetection2024!'
)

cur = conn.cursor()

print("📊 Database Table Status:")
print("=" * 50)

# First check what tables actually exist
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name
""")

existing_tables = [row[0] for row in cur.fetchall()]
print(f"Found {len(existing_tables)} tables in database:")
for table in existing_tables:
    print(f"  - {table}")

print("\n📈 Table Row Counts:")
print("=" * 50)

# Check row counts for existing tables
for table in existing_tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = cur.fetchone()[0]
        status = "✅ Has data" if count > 0 else "❌ Empty"
        print(f"{table:25} | {count:8} rows | {status}")
    except Exception as e:
        print(f"{table:25} | ERROR: {str(e)}")

# Check transactions specifically
if 'transactions' in existing_tables:
    print("\n🔍 Transaction Analysis:")
    print("=" * 50)
    
    try:
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN is_fraud = true THEN 1 END) as fraud_count,
                COUNT(CASE WHEN is_fraud = false THEN 1 END) as normal_count,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM transactions
            LIMIT 1
        """)
        
        result = cur.fetchone()
        if result and result[0] > 0:
            total, fraud, normal, oldest, newest = result
            print(f"Total transactions: {total:,}")
            print(f"Fraud transactions: {fraud:,} ({fraud/total*100:.1f}%)")
            print(f"Normal transactions: {normal:,} ({normal/total*100:.1f}%)")
            print(f"Date range: {oldest} to {newest}")
        else:
            print("No transactions found")
    except Exception as e:
        print(f"Error analyzing transactions: {e}")

conn.close()
print("\n✅ Database check completed")