#!/usr/bin/env python3
"""
Test CSV upload functionality
"""

import pandas as pd
from csv_fraud_processor import CSVFraudProcessor
import tempfile
import shutil
from datetime import datetime
import os

def test_csv_upload():
    print("🧪 Testing CSV Upload Functionality")
    print("=" * 40)
    
    # Create test data
    test_data = {
        'transaction_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004', 'tx_005'],
        'user_id': ['user_001', 'user_002', 'user_003', 'user_004', 'user_005'],
        'amount': [99.99, 5500.00, 25.50, 15000.00, 0.01],
        'merchant_id': ['merchant_normal', 'merchant_gambling', 'merchant_grocery', 'merchant_suspicious', 'merchant_test'],
        'category': ['electronics', 'gambling', 'grocery', 'crypto', 'test'],
        'currency': ['USD', 'USD', 'USD', 'USD', 'USD']
    }
    
    df = pd.DataFrame(test_data)
    
    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name
    
    print(f"✅ Created test CSV: {tmp_path}")
    print(f"📊 Data shape: {df.shape}")
    
    try:
        # Test CSV processor
        processor = CSVFraudProcessor()
        print("✅ CSV Processor initialized")
        
        # Read the file (simulating upload)
        df_read = pd.read_csv(tmp_path)
        print(f"✅ File read successfully: {len(df_read)} rows")
        
        # Process the data
        print("🔍 Processing data...")
        df_processed = processor.process_batch(df_read)
        print(f"✅ Processing completed: {len(df_processed)} rows")
        
        # Generate summary
        print("📊 Generating summary...")
        summary = processor.generate_summary_report(df_processed)
        print("✅ Summary generated")
        
        # Save results
        print("💾 Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"test_upload_{timestamp}"
        csv_path, json_path = processor.save_results(df_processed, summary, filename_prefix)
        print(f"✅ Results saved:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        # Print summary
        print("\n📊 PROCESSING SUMMARY")
        print("=" * 30)
        print(f"Total Transactions: {summary['total_transactions']}")
        print(f"Approved: {summary['decisions']['approved']}")
        print(f"Declined: {summary['decisions']['declined']}")
        print(f"Review: {summary['decisions']['review']}")
        print(f"Fraud Rate: {summary['fraud_rate']:.2%}")
        
        print("\n🎯 RISK LEVELS")
        print("=" * 20)
        for level, count in summary['risk_levels'].items():
            print(f"{level.upper()}: {count}")
        
        print("\n⚠️ TOP RISK FACTORS")
        print("=" * 25)
        for factor, count in summary['top_risk_factors'].items():
            print(f"{factor}: {count}")
        
        print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"🧹 Cleaned up temporary file: {tmp_path}")

if __name__ == "__main__":
    test_csv_upload()