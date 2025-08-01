#!/usr/bin/env python3
"""
Test script to verify CSV analytics functionality
"""

import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.advanced_analytics import AdvancedFraudAnalytics

def test_csv_analytics():
    """Test CSV analytics with demo data"""
    
    print("🔍 Testing CSV Analytics with Demo Data")
    print("=" * 50)
    
    # Load the demo CSV
    try:
        df = pd.read_csv('demo_transactions.csv')
        print(f"✅ Successfully loaded {len(df)} transactions")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"🕐 Sample timestamps: {df['timestamp'].iloc[:3].tolist()}")
        
        # Test timestamp parsing
        print("\n🔧 Testing timestamp parsing...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if df['timestamp'].isna().any():
            failed = df['timestamp'].isna().sum()
            print(f"⚠️ {failed} timestamps failed to parse")
        else:
            print("✅ All timestamps parsed successfully")
        
        # Test temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"⏰ Hour range: {df['hour'].min()}:00 to {df['hour'].max()}:00")
        
        # Test hourly analysis
        hourly_stats = df.groupby('hour').agg({
            'transaction_id': 'count',
            'fraud_score': 'mean',
            'is_fraud': 'sum'
        }).reset_index()
        
        print(f"\n📊 Hourly Analysis Results:")
        print(hourly_stats.head())
        
        # Test fraud rate calculation
        total_fraud = df['is_fraud'].sum()
        fraud_rate = (total_fraud / len(df)) * 100
        print(f"\n🚨 Fraud Statistics:")
        print(f"   Total transactions: {len(df)}")
        print(f"   Fraud cases: {total_fraud}")
        print(f"   Fraud rate: {fraud_rate:.2f}%")
        
        print("\n✅ CSV Analytics Test Complete!")
        print("\n💡 To use this data in the Streamlit app:")
        print("   1. Start the app: streamlit run app/main.py")
        print("   2. Go to 'CSV Processor' tab")
        print("   3. Upload 'demo_transactions.csv'")
        print("   4. Go to 'Advanced Analytics' tab")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_csv_analytics()