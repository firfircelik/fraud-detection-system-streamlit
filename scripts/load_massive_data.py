#!/usr/bin/env python3
"""
Load MASSIVE real transaction data into PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import logging
from datetime import datetime, timezone
import numpy as np
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_massive_data():
    """Load the massive transaction data"""
    
    # Connect to database (inside container)
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='fraud_detection',
        user='fraud_admin',
        password='FraudDetection2024!'
    )
    
    logger.info("🚀 Loading MASSIVE transaction data...")
    
    # Read CSV file
    csv_file = '/tmp/5M_transactions.csv'
    logger.info(f"Reading {csv_file}...")
    
    # Read in chunks to handle large file
    chunk_size = 50000
    total_rows = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        logger.info(f"Processing chunk {chunk_num + 1} with {len(chunk)} rows...")
        
        # Prepare data
        users_data = []
        merchants_data = []
        transactions_data = []
        
        # Get unique users and merchants from this chunk
        unique_users = chunk['user_id'].unique()
        unique_merchants = chunk['merchant_id'].unique()
        
        # Prepare users
        for user_id in unique_users:
            users_data.append((
                user_id,
                f"{user_id.lower()}@example.com",
                f"User_{user_id[-8:]}",
                f"Last_{user_id[-4:]}",
                f"+1{np.random.randint(1000000000, 9999999999)}",
                datetime.now(timezone.utc),
                'ACTIVE',
                np.random.uniform(0.0, 1.0),
                datetime.now(timezone.utc)
            ))
        
        # Prepare merchants
        for merchant_id in unique_merchants:
            category = chunk[chunk['merchant_id'] == merchant_id]['category'].iloc[0]
            merchants_data.append((
                merchant_id,
                f"Business_{merchant_id}",
                category.upper(),
                np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA']),
                f"City_{merchant_id[-6:]}",
                np.random.uniform(0.0, 0.5),
                np.random.uniform(0.0, 0.1),
                0,
                0.0,
                datetime.now(timezone.utc)
            ))
        
        # Prepare transactions
        for _, row in chunk.iterrows():
            # Parse timestamp
            try:
                timestamp = pd.to_datetime(row['timestamp'])
            except:
                timestamp = datetime.now(timezone.utc)
            
            # Determine fraud info
            is_fraud = bool(int(row['is_fraud']))
            fraud_score = np.random.uniform(0.8, 1.0) if is_fraud else np.random.uniform(0.0, 0.4)
            
            if fraud_score >= 0.8:
                risk_level = 'CRITICAL'
                decision = 'DECLINED'
            elif fraud_score >= 0.6:
                risk_level = 'HIGH'
                decision = 'REVIEW'
            elif fraud_score >= 0.4:
                risk_level = 'MEDIUM'
                decision = 'APPROVED'
            else:
                risk_level = 'LOW'
                decision = 'APPROVED'
            
            transactions_data.append((
                row['transaction_id'],
                row['user_id'],
                row['merchant_id'],
                float(row['amount']),
                row['currency'],
                'PURCHASE',
                np.random.choice(['CREDIT_CARD', 'DEBIT_CARD', 'PAYPAL', 'BANK_TRANSFER']),
                timestamp,
                float(row['lat']) if pd.notna(row['lat']) else None,
                float(row['lon']) if pd.notna(row['lon']) else None,
                np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA']),
                f"City_{np.random.randint(1, 1000)}",
                row['ip_address'],
                row['device_id'],
                np.random.choice(['MOBILE', 'DESKTOP', 'TABLET']),
                fraud_score,
                risk_level,
                is_fraud,
                decision,
                np.random.uniform(0.7, 1.0),
                np.random.randint(10, 500),
                'v2.0.0',
                'COMPLETED',
                datetime.now(timezone.utc)
            ))
        
        # Insert data
        with conn.cursor() as cur:
            # Insert users (ignore duplicates)
            if users_data:
                execute_batch(cur, """
                    INSERT INTO users (user_id, email, first_name, last_name, phone, 
                                     registration_date, account_status, risk_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                """, users_data, page_size=1000)
            
            # Insert merchants (ignore duplicates)
            if merchants_data:
                execute_batch(cur, """
                    INSERT INTO merchants (merchant_id, merchant_name, category, country, city,
                                         risk_score, fraud_rate, transaction_count, total_volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (merchant_id) DO NOTHING
                """, merchants_data, page_size=1000)
            
            # Insert transactions
            if transactions_data:
                execute_batch(cur, """
                    INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency,
                                            transaction_type, payment_method, transaction_timestamp,
                                            latitude, longitude, country, city, ip_address, device_id,
                                            device_type, fraud_score, risk_level, is_fraud, decision,
                                            confidence_score, processing_time_ms, model_version, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                """, transactions_data, page_size=1000)
            
            conn.commit()
            total_rows += len(chunk)
            logger.info(f"✅ Loaded chunk {chunk_num + 1}: {len(transactions_data)} transactions")
    
    # Update statistics
    logger.info("Updating merchant statistics...")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE merchants SET 
                transaction_count = (
                    SELECT COUNT(*) FROM transactions 
                    WHERE transactions.merchant_id = merchants.merchant_id
                ),
                total_volume = (
                    SELECT COALESCE(SUM(amount), 0) FROM transactions 
                    WHERE transactions.merchant_id = merchants.merchant_id
                ),
                fraud_rate = (
                    SELECT COALESCE(
                        COUNT(*) FILTER (WHERE is_fraud = true)::DECIMAL / 
                        NULLIF(COUNT(*), 0), 0
                    ) FROM transactions 
                    WHERE transactions.merchant_id = merchants.merchant_id
                )
        """)
        
        cur.execute("""
            UPDATE users SET 
                risk_score = (
                    SELECT COALESCE(AVG(fraud_score), 0) FROM transactions 
                    WHERE transactions.user_id = users.user_id
                )
        """)
        
        conn.commit()
    
    # Final stats
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM merchants")
        merchant_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM transactions")
        transaction_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = true")
        fraud_count = cur.fetchone()[0]
        
        fraud_rate = (fraud_count / transaction_count * 100) if transaction_count > 0 else 0
        
        logger.info(f"""
        🎉 MASSIVE DATA LOADED SUCCESSFULLY!
        ====================================
        👥 Users: {user_count:,}
        🏪 Merchants: {merchant_count:,}
        💳 Transactions: {transaction_count:,}
        🚨 Fraud Cases: {fraud_count:,} ({fraud_rate:.2f}%)
        ====================================
        """)
    
    conn.close()
    logger.info("✅ Data loading completed!")

if __name__ == "__main__":
    load_massive_data()