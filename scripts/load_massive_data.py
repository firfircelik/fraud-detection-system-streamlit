#!/usr/bin/env python3
"""
🚨 Massive Data Loader for Fraud Detection System
Loads large CSV files into PostgreSQL database efficiently
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassiveDataLoader:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.batch_size = int(os.getenv('BATCH_SIZE', 10000))
        self.data_dir = '/app/data'
        
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        self.engine = create_engine(self.database_url, pool_size=20, max_overflow=30)
        logger.info(f"Initialized data loader with batch size: {self.batch_size}")
    
    def load_users_and_merchants(self):
        """Load users and merchants from CSV data"""
        logger.info("Loading users and merchants...")
        
        try:
            # Sample CSV to extract unique users and merchants
            sample_file = os.path.join(self.data_dir, 'massive', '1M_transactions.csv')
            if not os.path.exists(sample_file):
                logger.warning(f"Sample file not found: {sample_file}")
                return
            
            # Read sample to get unique values
            logger.info("Reading sample data to extract users and merchants...")
            df_sample = pd.read_csv(sample_file, nrows=100000)
            
            # Extract unique users
            unique_users = df_sample['user_id'].unique()
            logger.info(f"Found {len(unique_users)} unique users")
            
            # Create users dataframe
            users_data = []
            for user_id in tqdm(unique_users, desc="Processing users"):
                users_data.append({
                    'user_id': user_id,
                    'email': f"{user_id.lower()}@example.com",
                    'first_name': f"User_{user_id.split('_')[-1]}",
                    'last_name': "Doe",
                    'country_code': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], p=[0.4, 0.2, 0.15, 0.15, 0.1]),
                    'risk_profile': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.7, 0.25, 0.05])
                })
            
            users_df = pd.DataFrame(users_data)
            
            # Insert users in batches
            with self.engine.connect() as conn:
                users_df.to_sql('users', conn, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(users_df)} users")
            
            # Extract unique merchants
            unique_merchants = df_sample['merchant_id'].unique()
            logger.info(f"Found {len(unique_merchants)} unique merchants")
            
            # Create merchants dataframe
            merchants_data = []
            categories = ['grocery', 'electronics', 'clothing', 'restaurant', 'gas', 'travel', 'entertainment', 'gambling', 'cryptocurrency']
            
            for merchant_id in tqdm(unique_merchants, desc="Processing merchants"):
                category = np.random.choice(categories)
                risk_score = 0.9 if category in ['gambling', 'cryptocurrency'] else np.random.uniform(0.1, 0.5)
                
                merchants_data.append({
                    'merchant_id': merchant_id,
                    'merchant_name': f"Merchant {merchant_id.split('_')[-1]}",
                    'business_type': 'Online' if np.random.random() > 0.3 else 'Physical',
                    'category': category,
                    'country': np.random.choice(['US', 'UK', 'CA']),
                    'risk_score': round(risk_score, 4)
                })
            
            merchants_df = pd.DataFrame(merchants_data)
            
            # Insert merchants in batches
            with self.engine.connect() as conn:
                merchants_df.to_sql('merchants', conn, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(merchants_df)} merchants")
            
        except Exception as e:
            logger.error(f"Error loading users and merchants: {e}")
            raise
    
    def load_transactions_from_csv(self, csv_file: str, max_rows: int = None):
        """Load transactions from CSV file efficiently"""
        logger.info(f"Loading transactions from {csv_file}")
        
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            return
        
        try:
            # Get file size for progress tracking
            file_size = os.path.getsize(csv_file)
            logger.info(f"File size: {file_size / (1024*1024):.1f} MB")
            
            # Read CSV in chunks
            chunk_iter = pd.read_csv(csv_file, chunksize=self.batch_size)
            
            total_processed = 0
            start_time = time.time()
            
            for chunk_num, chunk in enumerate(chunk_iter):
                if max_rows and total_processed >= max_rows:
                    break
                
                # Process chunk
                processed_chunk = self.process_transaction_chunk(chunk)
                
                # Insert to database
                with self.engine.connect() as conn:
                    processed_chunk.to_sql('transactions', conn, if_exists='append', index=False, method='multi')
                
                total_processed += len(processed_chunk)
                
                # Progress logging
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {total_processed:,} transactions ({rate:.0f} tx/sec)")
                
                # Memory cleanup
                del chunk, processed_chunk
            
            logger.info(f"Successfully loaded {total_processed:,} transactions from {csv_file}")
            
        except Exception as e:
            logger.error(f"Error loading transactions from {csv_file}: {e}")
            raise
    
    def process_transaction_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of transactions and add fraud detection results"""
        
        # Rename columns to match database schema
        column_mapping = {
            'timestamp': 'transaction_timestamp',
            'lat': 'latitude',
            'lon': 'longitude'
        }
        chunk = chunk.rename(columns=column_mapping)
        
        # Add missing columns
        chunk['transaction_status'] = 'COMPLETED'
        chunk['country'] = 'US'  # Default country
        chunk['processed_at'] = datetime.now()
        chunk['processing_time_ms'] = np.random.randint(50, 200)
        chunk['model_version'] = 'v1.0'
        chunk['confidence_score'] = np.random.uniform(0.7, 0.99, len(chunk))
        
        # Calculate fraud scores and risk levels
        fraud_scores = []
        risk_levels = []
        decisions = []
        
        for _, row in chunk.iterrows():
            # Simple fraud scoring logic
            score = 0.0
            
            # Amount-based scoring
            amount = float(row.get('amount', 0))
            if amount > 5000:
                score += 0.4
            elif amount > 1000:
                score += 0.2
            elif amount < 1:
                score += 0.3
            
            # Add some randomness for realistic distribution
            score += np.random.uniform(0, 0.3)
            score = min(1.0, max(0.0, score))
            
            # Determine risk level and decision
            if score >= 0.8:
                risk_level = 'CRITICAL'
                decision = 'DECLINED'
            elif score >= 0.6:
                risk_level = 'HIGH'
                decision = 'REVIEW'
            elif score >= 0.4:
                risk_level = 'MEDIUM'
                decision = 'REVIEW'
            elif score >= 0.2:
                risk_level = 'LOW'
                decision = 'APPROVED'
            else:
                risk_level = 'MINIMAL'
                decision = 'APPROVED'
            
            fraud_scores.append(round(score, 4))
            risk_levels.append(risk_level)
            decisions.append(decision)
        
        chunk['fraud_score'] = fraud_scores
        chunk['risk_level'] = risk_levels
        chunk['decision'] = decisions
        
        # Set is_fraud based on decision
        chunk['is_fraud'] = chunk['decision'] == 'DECLINED'
        
        return chunk
    
    def load_all_massive_data(self):
        """Load all available massive datasets"""
        massive_dir = os.path.join(self.data_dir, 'massive')
        
        if not os.path.exists(massive_dir):
            logger.warning(f"Massive data directory not found: {massive_dir}")
            return
        
        # Find CSV files
        csv_files = [f for f in os.listdir(massive_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Load users and merchants first
        self.load_users_and_merchants()
        
        # Process each CSV file
        for csv_file in sorted(csv_files):
            csv_path = os.path.join(massive_dir, csv_file)
            logger.info(f"Processing {csv_file}...")
            
            # Limit rows for demo (remove in production)
            max_rows = 50000 if '1M' in csv_file else None
            
            try:
                self.load_transactions_from_csv(csv_path, max_rows)
            except Exception as e:
                logger.error(f"Failed to process {csv_file}: {e}")
                continue
        
        # Update statistics
        self.update_statistics()
    
    def update_statistics(self):
        """Update database statistics and create indexes"""
        logger.info("Updating database statistics...")
        
        try:
            with self.engine.connect() as conn:
                # Update table statistics
                conn.execute(text("ANALYZE transactions"))
                conn.execute(text("ANALYZE users"))
                conn.execute(text("ANALYZE merchants"))
                
                # Get final counts
                tx_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
                fraud_count = conn.execute(text("SELECT COUNT(*) FROM transactions WHERE is_fraud = true")).scalar()
                user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                merchant_count = conn.execute(text("SELECT COUNT(*) FROM merchants")).scalar()
                
                logger.info(f"Final statistics:")
                logger.info(f"  Transactions: {tx_count:,}")
                logger.info(f"  Fraud cases: {fraud_count:,}")
                logger.info(f"  Users: {user_count:,}")
                logger.info(f"  Merchants: {merchant_count:,}")
                logger.info(f"  Fraud rate: {(fraud_count/tx_count*100):.2f}%")
                
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

def main():
    """Main function"""
    logger.info("🚨 Starting Massive Data Loader for Fraud Detection System")
    
    try:
        loader = MassiveDataLoader()
        loader.load_all_massive_data()
        logger.info("✅ Data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        raise

if __name__ == "__main__":
    main()