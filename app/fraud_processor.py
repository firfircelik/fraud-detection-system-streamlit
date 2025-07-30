#!/usr/bin/env python3
"""
🚨 CSV Fraud Detection Processor
Batch processing service for CSV files
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVFraudProcessor:
    """CSV Fraud Detection Processor"""
    
    def __init__(self):
        # Set data directory relative to current working directory
        self.data_dir = os.path.join(os.getcwd(), "data")
        self.massive_dir = os.path.join(self.data_dir, "massive")
        self.results_dir = os.path.join(self.data_dir, "results")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Fraud detection rules
        self.fraud_rules = {
            'high_amount_threshold': 1000.0,
            'very_high_amount_threshold': 5000.0,
            'suspicious_merchants': [
                'gambling', 'casino', 'crypto', 'bitcoin', 'darkweb', 
                'suspicious', 'unknown', 'offshore', 'adult', 'cash'
            ],
            'suspicious_categories': [
                'gambling', 'adult', 'crypto', 'cash_advance', 'atm', 'casino'
            ],
            'late_hour_start': 23,
            'early_hour_end': 6,
            'max_daily_transactions': 20,
            'velocity_threshold': 5  # transactions per hour
        }
    
    def load_csv_file(self, filename: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load CSV file with optional sampling and optimized chunking for large files"""
        try:
            filepath = os.path.join(self.massive_dir, filename)
            if not os.path.exists(filepath):
                filepath = os.path.join(self.data_dir, filename)
            
            logger.info(f"Loading CSV file: {filepath}")
            
            # Get file size for processing decision
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"File size: {file_size_mb:.1f} MB")
            
            if sample_size:
                # If sample size specified, just read that many rows
                df = pd.read_csv(filepath, nrows=sample_size)
                logger.info(f"Loaded {len(df)} rows (sampled)")
                return df
            
            # For large files (>100MB), use optimized chunked reading
            if file_size_mb > 100:
                logger.info(f"Large file detected ({file_size_mb:.1f}MB). Using optimized chunked reading...")
                
                # Read in larger chunks for better performance
                chunk_size = 50000  # Increased chunk size for large files
                chunks = []
                total_rows = 0
                
                for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Log progress for very large files
                    if len(chunks) % 20 == 0:  # Every 1M rows (20 * 50K)
                        logger.info(f"Processed {total_rows:,} rows...")
                
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded complete file: {len(df):,} rows")
                
            else:
                # For smaller files, read all at once
                df = pd.read_csv(filepath)
                logger.info(f"Loaded {len(df):,} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {filename}: {str(e)}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different CSV formats"""
        
        # Common column mappings
        column_mappings = {
            # Transaction ID variations
            'transaction_id': 'transaction_id',
            'TransactionID': 'transaction_id',
            'id': 'transaction_id',
            'ID': 'transaction_id',
            
            # User/Account ID variations
            'user_id': 'user_id',
            'UserID': 'user_id',
            'account_id': 'user_id',
            'AccountID': 'user_id',
            'customer_id': 'user_id',
            
            # Amount variations
            'amount': 'amount',
            'Amount': 'amount',
            'TransactionAmt': 'amount',
            'transaction_amount': 'amount',
            
            # Merchant variations
            'merchant_id': 'merchant_id',
            'MerchantID': 'merchant_id',
            'merchant': 'merchant_id',
            'Merchant': 'merchant_id',
            
            # Category variations
            'category': 'category',
            'Category': 'category',
            'merchant_category': 'category',
            'ProductCD': 'category',
            
            # Timestamp variations
            'timestamp': 'timestamp',
            'Timestamp': 'timestamp',
            'TransactionDT': 'timestamp',
            'transaction_time': 'timestamp',
            
            # Fraud label variations
            'is_fraud': 'is_fraud',
            'isFraud': 'is_fraud',
            'Class': 'is_fraud',
            'fraud': 'is_fraud'
        }
        
        # Rename columns
        df_renamed = df.rename(columns=column_mappings)
        
        # Add missing columns with defaults
        required_columns = {
            'transaction_id': lambda: [f"tx_{i:08d}" for i in range(len(df_renamed))],
            'user_id': lambda: [f"user_{i:06d}" for i in range(len(df_renamed))],
            'amount': lambda: np.random.uniform(10, 1000, len(df_renamed)),
            'merchant_id': lambda: [f"merchant_{i%1000:04d}" for i in range(len(df_renamed))],
            'category': lambda: np.random.choice(['grocery', 'electronics', 'gas', 'restaurant'], len(df_renamed)),
            'timestamp': lambda: [datetime.now().isoformat() for _ in range(len(df_renamed))],
            'currency': lambda: ['USD'] * len(df_renamed)
        }
        
        for col, default_func in required_columns.items():
            if col not in df_renamed.columns:
                df_renamed[col] = default_func()
        
        return df_renamed
    
    def calculate_fraud_score(self, row: pd.Series) -> Tuple[float, str, List[str]]:
        """Calculate fraud score for a single transaction"""
        
        score = 0.0
        risk_factors = []
        
        # Amount-based scoring
        amount = float(row.get('amount', 0))
        if amount > self.fraud_rules['very_high_amount_threshold']:
            score += 0.5
            risk_factors.append('very_high_amount')
        elif amount > self.fraud_rules['high_amount_threshold']:
            score += 0.35
            risk_factors.append('high_amount')
        elif amount > 2000:
            score += 0.2
            risk_factors.append('high_amount_moderate')
        elif amount < 1:
            score += 0.4
            risk_factors.append('micro_transaction')
        elif amount > 1000:
            score += 0.1
            risk_factors.append('elevated_amount')
        
        # Merchant-based scoring
        merchant = str(row.get('merchant_id', '')).lower()
        for suspicious_merchant in self.fraud_rules['suspicious_merchants']:
            if suspicious_merchant in merchant:
                score += 0.35
                risk_factors.append('suspicious_merchant')
                break
        
        # Category-based scoring
        category = str(row.get('category', '')).lower()
        if category in self.fraud_rules['suspicious_categories']:
            score += 0.3
            risk_factors.append('suspicious_category')
        
        # Time-based scoring (if timestamp available)
        try:
            timestamp = pd.to_datetime(row.get('timestamp'))
            hour = timestamp.hour
            if hour >= self.fraud_rules['late_hour_start'] or hour <= self.fraud_rules['early_hour_end']:
                score += 0.2
                risk_factors.append('unusual_hour')
        except:
            pass
        
        # Currency-based scoring
        currency = str(row.get('currency', 'USD')).upper()
        if currency not in ['USD', 'EUR', 'GBP']:
            score += 0.15
            risk_factors.append('unusual_currency')
        
        # Random factor for demo (simulating ML model uncertainty)
        random_factor = np.random.uniform(-0.05, 0.15)
        score += random_factor
        
        # Add some realistic fraud patterns
        user_id = str(row.get('user_id', ''))
        if 'test' in user_id.lower() or 'fake' in user_id.lower():
            score += 0.3
            risk_factors.append('suspicious_user_pattern')
        
        # Geographic risk (simplified)
        try:
            lat = float(row.get('lat', 0))
            lon = float(row.get('lon', 0))
            if abs(lat) > 80 or abs(lon) > 180:  # Invalid coordinates
                score += 0.2
                risk_factors.append('invalid_location')
        except:
            pass
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        # Determine risk level
        if score >= 0.8:
            risk_level = 'CRITICAL'
        elif score >= 0.6:
            risk_level = 'HIGH'
        elif score >= 0.4:
            risk_level = 'MEDIUM'
        elif score >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return score, risk_level, risk_factors
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of transactions"""
        
        logger.info(f"Processing batch of {len(df)} transactions")
        
        # Standardize columns
        df_processed = self.standardize_columns(df.copy())
        
        # Calculate fraud scores
        fraud_scores = []
        risk_levels = []
        risk_factors_list = []
        decisions = []
        
        for idx, row in df_processed.iterrows():
            score, risk_level, risk_factors = self.calculate_fraud_score(row)
            
            fraud_scores.append(score)
            risk_levels.append(risk_level)
            risk_factors_list.append(json.dumps(risk_factors))
            
            # Make decision
            if score >= 0.6:
                decision = 'DECLINED'
            elif score >= 0.35:
                decision = 'REVIEW'
            else:
                decision = 'APPROVED'
            
            decisions.append(decision)
        
        # Add results to dataframe
        df_processed['fraud_score'] = fraud_scores
        df_processed['risk_level'] = risk_levels
        df_processed['risk_factors'] = risk_factors_list
        df_processed['decision'] = decisions
        df_processed['processed_at'] = datetime.now().isoformat()
        
        logger.info(f"Batch processing completed")
        
        return df_processed
    
    def generate_summary_report(self, df_processed: pd.DataFrame) -> Dict:
        """Generate summary report for processed transactions"""
        
        total_transactions = len(df_processed)
        
        # Decision summary
        decision_counts = df_processed['decision'].value_counts().to_dict()
        
        # Risk level summary
        risk_counts = df_processed['risk_level'].value_counts().to_dict()
        
        # Fraud score statistics
        fraud_scores = df_processed['fraud_score']
        
        # Amount statistics
        amounts = pd.to_numeric(df_processed['amount'], errors='coerce')
        
        # Top risk factors
        all_risk_factors = []
        for factors_json in df_processed['risk_factors']:
            try:
                factors = json.loads(factors_json)
                all_risk_factors.extend(factors)
            except:
                pass
        
        risk_factor_counts = pd.Series(all_risk_factors).value_counts().head(10).to_dict()
        
        summary = {
            'total_transactions': total_transactions,
            'processing_timestamp': datetime.now().isoformat(),
            'decisions': {
                'approved': decision_counts.get('APPROVED', 0),
                'declined': decision_counts.get('DECLINED', 0),
                'review': decision_counts.get('REVIEW', 0)
            },
            'risk_levels': {
                'minimal': risk_counts.get('MINIMAL', 0),
                'low': risk_counts.get('LOW', 0),
                'medium': risk_counts.get('MEDIUM', 0),
                'high': risk_counts.get('HIGH', 0),
                'critical': risk_counts.get('CRITICAL', 0)
            },
            'fraud_score_stats': {
                'mean': float(fraud_scores.mean()),
                'median': float(fraud_scores.median()),
                'std': float(fraud_scores.std()),
                'min': float(fraud_scores.min()),
                'max': float(fraud_scores.max())
            },
            'amount_stats': {
                'mean': float(amounts.mean()),
                'median': float(amounts.median()),
                'total': float(amounts.sum()),
                'max': float(amounts.max())
            },
            'top_risk_factors': risk_factor_counts,
            'fraud_rate': len(df_processed[df_processed['decision'] == 'DECLINED']) / total_transactions
        }
        
        return summary
    
    def save_results(self, df_processed: pd.DataFrame, summary: Dict, filename_prefix: str) -> Tuple[str, str]:
        """Save processed results and summary"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed CSV
        csv_filename = f"{filename_prefix}_processed_{timestamp}.csv"
        csv_path = os.path.join(self.results_dir, csv_filename)
        df_processed.to_csv(csv_path, index=False)
        
        # Save summary JSON
        json_filename = f"{filename_prefix}_summary_{timestamp}.json"
        json_path = os.path.join(self.results_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved: {csv_path}, {json_path}")
        
        return csv_path, json_path
    
    def get_available_files(self) -> List[Dict]:
        """Get list of available CSV files"""
        
        files = []
        
        # Check data directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)
                size = os.path.getsize(filepath)
                files.append({
                    'name': filename,
                    'path': filepath,
                    'size': size,
                    'size_mb': round(size / (1024 * 1024), 2),
                    'location': 'data'
                })
        
        # Check massive directory
        if os.path.exists(self.massive_dir):
            for filename in os.listdir(self.massive_dir):
                if filename.endswith('.csv'):
                    filepath = os.path.join(self.massive_dir, filename)
                    size = os.path.getsize(filepath)
                    files.append({
                        'name': filename,
                        'path': filepath,
                        'size': size,
                        'size_mb': round(size / (1024 * 1024), 2),
                        'location': 'massive'
                    })
        
        return sorted(files, key=lambda x: x['size'], reverse=True)
    
    def process_file(self, filename: str, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict, str, str]:
        """Process a complete CSV file"""
        
        logger.info(f"Starting processing of file: {filename}")
        
        # Load data
        df = self.load_csv_file(filename, sample_size)
        
        # Process batch
        df_processed = self.process_batch(df)
        
        # Generate summary
        summary = self.generate_summary_report(df_processed)
        
        # Save results
        filename_prefix = filename.replace('.csv', '').replace('.', '_')
        csv_path, json_path = self.save_results(df_processed, summary, filename_prefix)
        
        logger.info(f"File processing completed: {filename}")
        
        return df_processed, summary, csv_path, json_path

# Example usage
if __name__ == "__main__":
    processor = CSVFraudProcessor()
    
    # Get available files
    files = processor.get_available_files()
    print("Available CSV files:")
    for file_info in files:
        print(f"  {file_info['name']} ({file_info['size_mb']} MB)")
    
    # Process a sample file
    if files:
        sample_file = files[0]['name']
        print(f"\nProcessing sample file: {sample_file}")
        
        df_processed, summary, csv_path, json_path = processor.process_file(
            sample_file, 
            sample_size=1000  # Process only 1000 rows for demo
        )
        
        print(f"\nProcessing Summary:")
        print(f"Total transactions: {summary['total_transactions']}")
        print(f"Approved: {summary['decisions']['approved']}")
        print(f"Declined: {summary['decisions']['declined']}")
        print(f"Review: {summary['decisions']['review']}")
        print(f"Fraud rate: {summary['fraud_rate']:.2%}")
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  Summary: {json_path}")