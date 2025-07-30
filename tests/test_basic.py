"""
Basic tests for fraud detection system
"""
import pytest
import pandas as pd
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

def test_imports():
    """Test that core modules can be imported"""
    try:
        from fraud_processor import CSVFraudProcessor
        from main import main
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_fraud_processor_init():
    """Test fraud processor initialization"""
    try:
        from fraud_processor import CSVFraudProcessor
        processor = CSVFraudProcessor()
        assert processor is not None
    except Exception as e:
        pytest.fail(f"Fraud processor init failed: {e}")

def test_basic_dataframe_processing():
    """Test basic dataframe processing"""
    try:
        from fraud_processor import CSVFraudProcessor
        
        # Create simple test data
        test_data = pd.DataFrame({
            'transaction_id': ['tx_001', 'tx_002'],
            'amount': [100.0, 500.0],
            'merchant_id': ['merchant_001', 'merchant_002']
        })
        
        processor = CSVFraudProcessor()
        result = processor.process_dataframe(test_data)
        
        assert result is not None
        assert len(result) == 2
        assert 'fraud_score' in result.columns
        
    except Exception as e:
        pytest.fail(f"Basic processing test failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_fraud_processor_init() 
    test_basic_dataframe_processing()
    print("✅ All tests passed!")
