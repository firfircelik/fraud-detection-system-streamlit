#!/usr/bin/env python3
"""
Script to replace all hardcoded psycopg2 connections with SQLAlchemy engine connections
"""

import re

def fix_psycopg2_connections(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match psycopg2 connection blocks
    pattern = r"""import psycopg2\s*
\s*conn = psycopg2\.connect\(\s*
\s*host='127\.0\.0\.1',\s*
\s*port=5432,\s*
\s*database='fraud_detection',\s*
\s*user='fraud_admin',\s*
\s*password='FraudDetection2024!'\s*
\s*\)"""
    
    replacement = """if engine is None:
                raise HTTPException(status_code=500, detail="Database connection not available")
            
            with engine.connect() as conn:"""
    
    # Replace the pattern
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Replace cursor usage patterns
    content = re.sub(r'with conn\.cursor\(\) as cur:', 'pass  # Using SQLAlchemy connection', content)
    content = re.sub(r'cur\.execute\(', 'result = conn.execute(text(', content)
    content = re.sub(r'cur\.fetchall\(\)', 'result.fetchall()', content)
    content = re.sub(r'cur\.fetchone\(\)', 'result.fetchone()', content)
    content = re.sub(r'conn\.close\(\)', '# Connection closed automatically by context manager', content)
    
    # Fix SQL query endings to add closing parenthesis for text()
    content = re.sub(r'text\("""([^"]+)"""\)', r'text("""\1"""))', content)
    content = re.sub(r'text\("([^"]+)"\)', r'text("\1"))', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed psycopg2 connections in {file_path}")

if __name__ == "__main__":
    fix_psycopg2_connections('/Users/firatcelik/Desktop/fraud-detection-system-streamlit/backend/api/main.py')
    print("All psycopg2 connections have been replaced with SQLAlchemy engine connections")