"""
Basic tests for fraud detection system - No pytest version
"""
import pandas as pd
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

def test_imports():
    """Test that core modules can be imported"""
    from fraud_processor import CSVFraudProcessor
    # Just test that modules exist, don't import main functions
    assert CSVFraudProcessor is not None
    print("✅ Import test passed")

def test_fraud_processor_init():
    """Test fraud processor initialization"""
    from fraud_processor import CSVFraudProcessor
    processor = CSVFraudProcessor()
    assert processor is not None
    print("✅ Fraud processor init test passed")

def test_basic_dataframe_processing():
    """Test basic dataframe processing"""
    from fraud_processor import CSVFraudProcessor
    
    # Create simple test data
    test_data = pd.DataFrame({
        'transaction_id': ['tx_001', 'tx_002'],
        'amount': [100.0, 500.0],
        'merchant_id': ['merchant_001', 'merchant_002']
    })
    
    processor = CSVFraudProcessor()
    result = processor.process_batch(test_data)  # Correct method name
    
    assert result is not None
    assert len(result) == 2
    assert 'fraud_score' in result.columns
    print("✅ Basic processing test passed")

if __name__ == "__main__":
    print("🧪 Running basic tests...")
    
    try:
        test_imports()
        test_fraud_processor_init() 
        test_basic_dataframe_processing()
        print("🎉 All tests passed!")
    except Exception as e:
        print(f"💥 Test failed: {e}")
        exit(1)
