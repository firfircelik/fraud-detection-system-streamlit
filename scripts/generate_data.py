#!/usr/bin/env python3
"""
MASSIVE Fraud Detection Dataset Generator
Creates synthetic datasets with 1M+ records for stress testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MassiveDatasetGenerator:
    def __init__(self, output_dir="data/massive"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration
        self.merchants = [f'MERCHANT_{i:08d}' for i in range(50000)]
        self.users = [f'USER_{i:09d}' for i in range(500000)]
        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']
        self.categories = [
            'grocery', 'electronics', 'clothing', 'restaurant', 'gas',
            'online_shopping', 'travel', 'entertainment', 'utilities', 'healthcare'
        ]
        self.countries = ['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU', 'BR', 'IN', 'MX']
        self.devices = [f'DEVICE_{i:010d}' for i in range(1000000)]
        
    def generate_transaction(self, is_fraud=False):
        """Generate a single transaction record"""
        base_amount = np.random.lognormal(3.5, 1.5)
        fraud_multiplier = np.random.uniform(2.0, 10.0) if is_fraud else 1.0
        amount = base_amount * fraud_multiplier
        
        # Fraud patterns
        if is_fraud:
            # Fraudulent transactions tend to be larger and at unusual times
            hour = np.random.choice([2, 3, 4, 5, 23], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            lat = np.random.uniform(-90, 90)  # Random global location
            lon = np.random.uniform(-180, 180)
        else:
            # Normal transactions follow business hours and realistic locations
            hour = np.random.choice(range(6, 23), p=np.ones(17)/17)
            # Cluster around major cities
            city_centers = [(40.7128, -74.0060), (51.5074, -0.1278), (35.6762, 139.6503)]
            center_lat, center_lon = np.random.choice(city_centers)
            lat = center_lat + np.random.normal(0, 2)
            lon = center_lon + np.random.normal(0, 2)
        
        timestamp = datetime.now() - timedelta(
            days=np.random.randint(0, 730),
            hours=hour,
            minutes=np.random.randint(0, 60)
        )
        
        # IP address generation
        octets = [np.random.randint(1, 255) for _ in range(4)]
        if is_fraud:
            # Fraudulent IPs often from specific ranges
            octets[0] = np.random.choice([41, 102, 197])  # Known fraud IP ranges
        
        return {
            'transaction_id': str(uuid.uuid4()),
            'user_id': np.random.choice(self.users),
            'amount': round(amount, 2),
            'currency': np.random.choice(self.currencies),
            'merchant_id': np.random.choice(self.merchants),
            'category': np.random.choice(self.categories),
            'country': np.random.choice(self.countries),
            'timestamp': timestamp.isoformat(),
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'device_id': np.random.choice(self.devices),
            'ip_address': '.'.join(map(str, octets)),
            'user_age': np.random.randint(18, 80),
            'user_income': np.random.choice(['low', 'medium', 'high', 'very_high']),
            'transaction_hour': hour,
            'transaction_day_of_week': timestamp.weekday(),
            'is_fraud': int(is_fraud)
        }
    
    def generate_dataset(self, n_records, fraud_rate=0.05, filename=None):
        """Generate a massive dataset"""
        if filename is None:
            filename = f'synthetic_{n_records//1000000}m_{int(fraud_rate*100)}pct.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"🚀 Generating {n_records:,} transactions...")
        print(f"📊 Fraud rate: {fraud_rate*100}%")
        print(f"💾 Output: {filepath}")
        
        # Calculate fraud vs normal
        n_fraud = int(n_records * fraud_rate)
        n_normal = n_records - n_fraud
        
        # Generate in batches for memory efficiency
        batch_size = 100000
        total_batches = (n_records + batch_size - 1) // batch_size
        
        with open(filepath, 'w') as f:
            # Write header
            sample_record = self.generate_transaction()
            header = ','.join(sample_record.keys())
            f.write(header + '\n')
            
            # Generate records in batches
            records_generated = 0
            
            # Generate fraud records
            for _ in tqdm(range(n_fraud), desc="Generating fraud records", unit="records"):
                record = self.generate_transaction(is_fraud=True)
                f.write(','.join(map(str, record.values())) + '\n')
                records_generated += 1
            
            # Generate normal records
            for _ in tqdm(range(n_normal), desc="Generating normal records", unit="records"):
                record = self.generate_transaction(is_fraud=False)
                f.write(','.join(map(str, record.values())) + '\n')
                records_generated += 1
        
        # Verify the dataset
        df = pd.read_csv(filepath, nrows=1000)
        actual_fraud_rate = df['is_fraud'].mean()
        
        print(f"✅ Dataset generated successfully!")
        print(f"📊 Total records: {records_generated:,}")
        print(f"🎯 Actual fraud rate: {actual_fraud_rate*100:.2f}%")
        print(f"💾 File size: {os.path.getsize(filepath) / 1024**2:.1f} MB")
        
        return filepath
    
    def generate_multiple_sizes(self):
        """Generate datasets of various sizes"""
        datasets = [
            (1000000, 0.05, "1M_transactions.csv"),
            (3000000, 0.03, "3M_transactions.csv"),
            (5000000, 0.02, "5M_transactions.csv"),
            (10000000, 0.01, "10M_transactions.csv"),
        ]
        
        generated_files = []
        for n_records, fraud_rate, filename in datasets:
            filepath = self.generate_dataset(n_records, fraud_rate, filename)
            generated_files.append(filepath)
        
        return generated_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate massive fraud detection datasets')
    parser.add_argument('--size', type=int, default=5000000, help='Number of records to generate')
    parser.add_argument('--fraud-rate', type=float, default=0.05, help='Fraud rate (0.0-1.0)')
    parser.add_argument('--output', type=str, default='data/massive', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Generate all dataset sizes')
    
    args = parser.parse_args()
    
    generator = MassiveDatasetGenerator(args.output)
    
    if args.all:
        generator.generate_multiple_sizes()
    else:
        generator.generate_dataset(args.size, args.fraud_rate)
    
    print("🎉 All datasets generated successfully!")