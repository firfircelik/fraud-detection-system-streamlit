#!/usr/bin/env python3
"""
Test script to verify timestamp format handling
"""

import pandas as pd
from datetime import datetime

# Test the exact timestamp format provided by user
test_timestamp = "2024-11-12T20:24:46.222183"

print("Testing timestamp format: 2024-11-12T20:24:46.222183")
print("=" * 50)

# Create test data with the exact format
test_data = {
    'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
    'user_id': ['USER123', 'USER456', 'USER789'],
    'amount': [150.50, 89.99, 250.00],
    'merchant_id': ['MERCHANT001', 'MERCHANT002', 'MERCHANT003'],
    'category': ['groceries', 'electronics', 'clothing'],
    'timestamp': [test_timestamp, 
                  "2024-11-12T21:15:32.445621", 
                  "2024-11-13T09:30:15.123456"],
    'fraud_score': [0.15, 0.85, 0.45]
}

df = pd.DataFrame(test_data)

print("Original DataFrame:")
print(df)
print(f"\nTimestamp column dtype: {df['timestamp'].dtype}")

# Test the parsing that will be used in the app
print("\n" + "=" * 50)
print("Testing timestamp parsing...")

try:
    # Parse datetime with timezone handling
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Check for parsing failures
    if df['timestamp'].isna().any():
        failed_count = df['timestamp'].isna().sum()
        print(f"⚠️ {failed_count} timestamp values could not be parsed")
    else:
        print("✅ All timestamps parsed successfully!")
    
    # Extract temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    
    print("\nParsed DataFrame:")
    print(df[['timestamp', 'hour', 'day_of_week', 'date', 'month']])
    
    # Test hourly analysis
    print("\n" + "=" * 50)
    print("Testing hourly analysis...")
    hourly_stats = df.groupby('hour')['fraud_score'].agg(['mean', 'count']).round(3)
    print(hourly_stats)
    
    print("\n✅ Timestamp format is fully supported!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("Expected CSV format:")
print("transaction_id,user_id,amount,merchant_id,category,timestamp,fraud_score")
print("TXN001,USER123,150.50,MERCHANT001,groceries,2024-11-12T20:24:46.222183,0.15")