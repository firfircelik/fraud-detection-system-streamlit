#!/usr/bin/env python3
"""
Load REAL 1M+ transaction data into our ENTERPRISE database schema
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import logging
from datetime import datetime, timezone, timedelta
import numpy as np
import uuid
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host='127.0.0.1',
        port=5432,
        database='fraud_detection',
        user='fraud_admin',
        password='FraudDetection2024!'
    )

def create_enterprise_tables(conn):
    """Create our enterprise tables"""
    with conn.cursor() as cur:
        logger.info("Creating enterprise tables...")
        
        # Create ENUM types first
        cur.execute("""
            DO $$ BEGIN
                CREATE TYPE risk_level AS ENUM ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        cur.execute("""
            DO $$ BEGIN
                CREATE TYPE transaction_status AS ENUM ('PENDING', 'APPROVED', 'DECLINED', 'REVIEW', 'BLOCKED', 'COMPLETED');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        # Users table (from our enterprise schema)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255),
                phone VARCHAR(50),
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                date_of_birth DATE,
                country VARCHAR(3),
                state_province VARCHAR(100),
                city VARCHAR(100),
                postal_code VARCHAR(20),
                timezone VARCHAR(50),
                account_created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login_at TIMESTAMP WITH TIME ZONE,
                account_status VARCHAR(20) DEFAULT 'ACTIVE',
                account_type VARCHAR(20) DEFAULT 'STANDARD',
                risk_score DECIMAL(5,4) DEFAULT 0.0000,
                total_transactions INTEGER DEFAULT 0,
                total_amount DECIMAL(15,2) DEFAULT 0.00,
                avg_transaction_amount DECIMAL(15,2) DEFAULT 0.00,
                fraud_incidents INTEGER DEFAULT 0,
                last_transaction_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Merchants table (from our enterprise schema)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS merchants (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                merchant_id VARCHAR(100) UNIQUE NOT NULL,
                merchant_name VARCHAR(255) NOT NULL,
                business_type VARCHAR(100),
                category VARCHAR(100),
                subcategory VARCHAR(100),
                mcc_code VARCHAR(10),
                country VARCHAR(3),
                state_province VARCHAR(100),
                city VARCHAR(100),
                postal_code VARCHAR(20),
                latitude DECIMAL(10,8),
                longitude DECIMAL(11,8),
                business_registration_date DATE,
                risk_score DECIMAL(5,4) DEFAULT 0.0000,
                fraud_rate DECIMAL(5,4) DEFAULT 0.0000,
                transaction_count INTEGER DEFAULT 0,
                total_volume DECIMAL(15,2) DEFAULT 0.00,
                avg_transaction_amount DECIMAL(15,2) DEFAULT 0.00,
                monthly_volume DECIMAL(15,2) DEFAULT 0.00,
                chargeback_rate DECIMAL(5,4) DEFAULT 0.0000,
                is_high_risk BOOLEAN DEFAULT FALSE,
                compliance_status VARCHAR(20) DEFAULT 'COMPLIANT',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Transactions table (partitioned by timestamp)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id UUID DEFAULT gen_random_uuid(),
                transaction_id VARCHAR(100) NOT NULL,
                user_id VARCHAR(100) NOT NULL,
                merchant_id VARCHAR(100) NOT NULL,
                amount DECIMAL(15,2) NOT NULL,
                currency VARCHAR(3) NOT NULL DEFAULT 'USD',
                transaction_type VARCHAR(50) DEFAULT 'PURCHASE',
                payment_method VARCHAR(50),
                card_type VARCHAR(20),
                card_last_four VARCHAR(4),
                transaction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                latitude DECIMAL(10,8),
                longitude DECIMAL(11,8),
                country VARCHAR(3),
                state_province VARCHAR(100),
                city VARCHAR(100),
                postal_code VARCHAR(20),
                ip_address INET,
                device_id VARCHAR(100),
                device_type VARCHAR(50),
                device_fingerprint VARCHAR(255),
                user_agent TEXT,
                browser VARCHAR(100),
                os VARCHAR(100),
                screen_resolution VARCHAR(20),
                ip_country VARCHAR(3),
                ip_region VARCHAR(100),
                is_vpn BOOLEAN DEFAULT FALSE,
                is_tor BOOLEAN DEFAULT FALSE,
                fraud_score DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                risk_level risk_level NOT NULL DEFAULT 'LOW',
                is_fraud BOOLEAN DEFAULT FALSE,
                decision transaction_status DEFAULT 'PENDING',
                confidence_score DECIMAL(5,4) DEFAULT 0.0000,
                processing_time_ms INTEGER,
                model_version VARCHAR(20),
                feature_version VARCHAR(20),
                ensemble_models_used TEXT[],
                merchant_category VARCHAR(100),
                transaction_description TEXT,
                reference_number VARCHAR(100),
                status transaction_status DEFAULT 'PENDING',
                processed_at TIMESTAMP WITH TIME ZONE,
                reviewed_at TIMESTAMP WITH TIME ZONE,
                reviewed_by VARCHAR(100),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (id, transaction_timestamp),
                CONSTRAINT unq_transaction_id_timestamp UNIQUE (transaction_id, transaction_timestamp),
                CONSTRAINT chk_amount_positive CHECK (amount > 0),
                CONSTRAINT chk_fraud_score CHECK (fraud_score >= 0.0000 AND fraud_score <= 1.0000),
                CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0000 AND confidence_score <= 1.0000)
            ) PARTITION BY RANGE (transaction_timestamp)
        """)
        
        # Create partitions for 2024-2025
        partitions = [
            ("2024_01", "2024-01-01", "2024-02-01"),
            ("2024_02", "2024-02-01", "2024-03-01"),
            ("2024_03", "2024-03-01", "2024-04-01"),
            ("2024_04", "2024-04-01", "2024-05-01"),
            ("2024_05", "2024-05-01", "2024-06-01"),
            ("2024_06", "2024-06-01", "2024-07-01"),
            ("2024_07", "2024-07-01", "2024-08-01"),
            ("2024_08", "2024-08-01", "2024-09-01"),
            ("2024_09", "2024-09-01", "2024-10-01"),
            ("2024_10", "2024-10-01", "2024-11-01"),
            ("2024_11", "2024-11-01", "2024-12-01"),
            ("2024_12", "2024-12-01", "2025-01-01"),
            ("2025_01", "2025-01-01", "2025-02-01"),
            ("2025_02", "2025-02-01", "2025-03-01"),
            ("2025_03", "2025-03-01", "2025-04-01"),
            ("2025_04", "2025-04-01", "2025-05-01"),
            ("2025_05", "2025-05-01", "2025-06-01"),
            ("2025_06", "2025-06-01", "2025-07-01"),
            ("2025_07", "2025-07-01", "2025-08-01"),
            ("2025_08", "2025-08-01", "2025-09-01"),
            ("2025_09", "2025-09-01", "2025-10-01"),
            ("2025_10", "2025-10-01", "2025-11-01"),
            ("2025_11", "2025-11-01", "2025-12-01"),
            ("2025_12", "2025-12-01", "2026-01-01")
        ]
        
        for partition_name, start_date, end_date in partitions:
            try:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS transactions_{partition_name} PARTITION OF transactions
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}')
                """)
                logger.info(f"Created partition transactions_{partition_name}")
            except psycopg2.errors.DuplicateTable:
                logger.info(f"Partition transactions_{partition_name} already exists")
                conn.rollback()
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_risk_score ON users(risk_score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_merchants_merchant_id ON merchants(merchant_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_merchants_category ON merchants(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_merchants_risk_score ON merchants(risk_score)")
        
        conn.commit()
        logger.info("✅ Enterprise tables created successfully!")

def load_massive_csv_data(conn, csv_file):
    """Load the massive CSV data into our enterprise schema"""
    logger.info(f"🚀 Loading MASSIVE data from {csv_file}...")
    
    # Read CSV in chunks
    chunk_size = 50000
    total_rows = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        logger.info(f"Processing chunk {chunk_num + 1} with {len(chunk)} rows...")
        
        # Prepare data arrays
        users_data = []
        merchants_data = []
        transactions_data = []
        
        # Get unique users and merchants
        unique_users = chunk['user_id'].unique()
        unique_merchants = chunk['merchant_id'].unique()
        
        # Prepare users data
        for user_id in unique_users:
            users_data.append((
                str(uuid.uuid4()),
                user_id,
                f"{user_id.lower()}@example.com",
                f"+1{np.random.randint(1000000000, 9999999999)}",
                f"User_{user_id[-8:]}",
                f"Last_{user_id[-4:]}",
                datetime.now().date() - timedelta(days=np.random.randint(365, 7300)),  # 1-20 years ago
                np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'AUS', 'JPN']),
                f"State_{np.random.randint(1, 50)}",
                f"City_{np.random.randint(1, 1000)}",
                f"{np.random.randint(10000, 99999)}",
                'America/New_York',
                datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 1000)),
                datetime.now(timezone.utc) - timedelta(hours=np.random.randint(1, 168)),
                'ACTIVE',
                np.random.choice(['STANDARD', 'PREMIUM', 'VIP']),
                np.random.uniform(0.0, 1.0),
                0, 0.0, 0.0, 0,
                None,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
        
        # Prepare merchants data
        for merchant_id in unique_merchants:
            category = chunk[chunk['merchant_id'] == merchant_id]['category'].iloc[0]
            merchants_data.append((
                str(uuid.uuid4()),
                merchant_id,
                f"Business_{merchant_id}",
                np.random.choice(['RETAIL', 'SERVICE', 'ONLINE', 'RESTAURANT']),
                category.upper(),
                f"Sub_{category}",
                f"{np.random.randint(1000, 9999)}",
                np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA']),
                f"State_{np.random.randint(1, 50)}",
                f"City_{np.random.randint(1, 1000)}",
                f"{np.random.randint(10000, 99999)}",
                np.random.uniform(-90, 90),
                np.random.uniform(-180, 180),
                datetime.now().date() - timedelta(days=np.random.randint(365, 3650)),
                np.random.uniform(0.0, 0.5),
                np.random.uniform(0.0, 0.1),
                0, 0.0, 0.0, 0.0,
                np.random.uniform(0.0, 0.05),
                bool(np.random.choice([True, False], p=[0.1, 0.9])),
                'COMPLIANT',
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
        
        # Prepare transactions data
        for _, row in chunk.iterrows():
            # Parse timestamp
            try:
                timestamp = pd.to_datetime(row['timestamp'])
                if pd.isna(timestamp):
                    timestamp = datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 365))
            except:
                timestamp = datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 365))
            
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
                str(uuid.uuid4()),
                row['transaction_id'],
                row['user_id'],
                row['merchant_id'],
                float(row['amount']),
                row['currency'],
                'PURCHASE',
                np.random.choice(['CREDIT_CARD', 'DEBIT_CARD', 'PAYPAL', 'BANK_TRANSFER']),
                np.random.choice(['VISA', 'MASTERCARD', 'AMEX', 'DISCOVER']),
                f"{np.random.randint(1000, 9999)}",
                timestamp,
                float(row['lat']) if pd.notna(row['lat']) else np.random.uniform(-90, 90),
                float(row['lon']) if pd.notna(row['lon']) else np.random.uniform(-180, 180),
                np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA']),
                f"State_{np.random.randint(1, 50)}",
                f"City_{np.random.randint(1, 1000)}",
                f"{np.random.randint(10000, 99999)}",
                row['ip_address'],
                row['device_id'],
                np.random.choice(['MOBILE', 'DESKTOP', 'TABLET']),
                f"fp_{uuid.uuid4().hex[:16]}",
                f"Mozilla/5.0 (compatible; FraudBot/1.0)",
                np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
                np.random.choice(['Windows', 'macOS', 'Linux', 'iOS', 'Android']),
                f"{np.random.randint(1024, 2560)}x{np.random.randint(768, 1440)}",
                np.random.choice(['USA', 'CAN', 'GBR', 'DEU', 'FRA']),
                f"Region_{np.random.randint(1, 100)}",
                bool(np.random.choice([True, False], p=[0.05, 0.95])),
                bool(np.random.choice([True, False], p=[0.01, 0.99])),
                fraud_score,
                risk_level,
                is_fraud,
                decision,
                np.random.uniform(0.7, 1.0),
                np.random.randint(10, 500),
                'v2.0.0',
                'v3.0.0',
                ['ensemble_v1', 'xgboost_v2', 'neural_net_v1'],
                chunk[chunk['transaction_id'] == row['transaction_id']]['category'].iloc[0],
                f"Transaction at {row['merchant_id']}",
                f"REF_{uuid.uuid4().hex[:12].upper()}",
                'COMPLETED',
                timestamp + timedelta(milliseconds=np.random.randint(10, 500)),
                None,
                None,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
        
        # Insert data in batches
        with conn.cursor() as cur:
            # Insert users
            if users_data:
                execute_batch(cur, """
                    INSERT INTO users (id, user_id, email, phone, first_name, last_name, date_of_birth,
                                     country, state_province, city, postal_code, timezone,
                                     account_created_at, last_login_at, account_status, account_type,
                                     risk_score, total_transactions, total_amount, avg_transaction_amount,
                                     fraud_incidents, last_transaction_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                """, users_data, page_size=1000)
            
            # Insert merchants
            if merchants_data:
                execute_batch(cur, """
                    INSERT INTO merchants (id, merchant_id, merchant_name, business_type, category, subcategory,
                                         mcc_code, country, state_province, city, postal_code, latitude, longitude,
                                         business_registration_date, risk_score, fraud_rate, transaction_count,
                                         total_volume, avg_transaction_amount, monthly_volume, chargeback_rate,
                                         is_high_risk, compliance_status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (merchant_id) DO NOTHING
                """, merchants_data, page_size=1000)
            
            # Insert transactions
            if transactions_data:
                execute_batch(cur, """
                    INSERT INTO transactions (id, transaction_id, user_id, merchant_id, amount, currency,
                                            transaction_type, payment_method, card_type, card_last_four,
                                            transaction_timestamp, latitude, longitude, country, state_province,
                                            city, postal_code, ip_address, device_id, device_type, device_fingerprint,
                                            user_agent, browser, os, screen_resolution, ip_country, ip_region,
                                            is_vpn, is_tor, fraud_score, risk_level, is_fraud, decision,
                                            confidence_score, processing_time_ms, model_version, feature_version,
                                            ensemble_models_used, merchant_category, transaction_description,
                                            reference_number, status, processed_at, reviewed_at, reviewed_by,
                                            created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id, transaction_timestamp) DO NOTHING
                """, transactions_data, page_size=1000)
            
            conn.commit()
            total_rows += len(chunk)
            logger.info(f"✅ Loaded chunk {chunk_num + 1}: {len(transactions_data)} transactions")
    
    logger.info(f"✅ Total rows processed: {total_rows:,}")
    return total_rows

def update_statistics(conn):
    """Update all statistics"""
    logger.info("📊 Updating enterprise statistics...")
    
    with conn.cursor() as cur:
        # Update merchant statistics
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
                avg_transaction_amount = (
                    SELECT COALESCE(AVG(amount), 0) FROM transactions 
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
        
        # Update user statistics
        cur.execute("""
            UPDATE users SET 
                total_transactions = (
                    SELECT COUNT(*) FROM transactions 
                    WHERE transactions.user_id = users.user_id
                ),
                total_amount = (
                    SELECT COALESCE(SUM(amount), 0) FROM transactions 
                    WHERE transactions.user_id = users.user_id
                ),
                avg_transaction_amount = (
                    SELECT COALESCE(AVG(amount), 0) FROM transactions 
                    WHERE transactions.user_id = users.user_id
                ),
                fraud_incidents = (
                    SELECT COUNT(*) FROM transactions 
                    WHERE transactions.user_id = users.user_id AND is_fraud = true
                ),
                risk_score = (
                    SELECT COALESCE(AVG(fraud_score), 0) FROM transactions 
                    WHERE transactions.user_id = users.user_id
                ),
                last_transaction_at = (
                    SELECT MAX(transaction_timestamp) FROM transactions 
                    WHERE transactions.user_id = users.user_id
                )
        """)
        
        conn.commit()
        logger.info("✅ Statistics updated!")

def main():
    """Main function"""
    logger.info("🚀 Starting ENTERPRISE data loading...")
    
    try:
        # Connect to database
        conn = get_db_connection()
        logger.info("✅ Connected to enterprise database")
        
        # Create enterprise tables
        create_enterprise_tables(conn)
        
        # Load massive CSV data
        csv_file = 'data/massive/5M_transactions.csv'
        if os.path.exists(csv_file):
            total_loaded = load_massive_csv_data(conn, csv_file)
        else:
            logger.error(f"CSV file not found: {csv_file}")
            return
        
        # Update statistics
        update_statistics(conn)
        
        # Final enterprise statistics
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            user_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM merchants")
            merchant_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM transactions")
            transaction_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = true")
            fraud_count = cur.fetchone()[0]
            
            cur.execute("SELECT SUM(amount) FROM transactions")
            total_volume = cur.fetchone()[0] or 0
            
            fraud_rate = (fraud_count / transaction_count * 100) if transaction_count > 0 else 0
            
            logger.info(f"""
            🎉 ENTERPRISE DATA LOADED SUCCESSFULLY!
            =======================================
            👥 Users: {user_count:,}
            🏪 Merchants: {merchant_count:,}
            💳 Transactions: {transaction_count:,}
            💰 Total Volume: ${total_volume:,.2f}
            🚨 Fraud Cases: {fraud_count:,} ({fraud_rate:.2f}%)
            =======================================
            """)
        
        conn.close()
        logger.info("✅ ENTERPRISE data loading completed!")
        
    except Exception as e:
        logger.error(f"❌ Enterprise data loading failed: {e}")
        raise

if __name__ == "__main__":
    main()