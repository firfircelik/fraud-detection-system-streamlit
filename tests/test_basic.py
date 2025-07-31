#!/usr/bin/env python3
"""
Basic tests for fraud detection system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import sqlite3
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_pandas_operations():
    """Test basic pandas operations"""
    df = pd.DataFrame({
        'transaction_id': ['tx_001', 'tx_002', 'tx_003'],
        'amount': [100.0, 200.0, 300.0],
        'merchant_id': ['merchant_1', 'merchant_2', 'merchant_3']
    })
    
    assert len(df) == 3
    assert 'transaction_id' in df.columns
    assert df['amount'].sum() == 600.0

def test_numpy_operations():
    """Test basic numpy operations"""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15

def test_fraud_score_calculation():
    """Test basic fraud score calculation logic"""
    # Simple fraud score calculation
    def calculate_simple_fraud_score(amount, merchant_type):
        score = 0.0
        
        if amount > 1000:
            score += 0.3
        
        if merchant_type in ['casino', 'gambling']:
            score += 0.5
        
        return min(score, 1.0)
    
    # Test cases
    assert calculate_simple_fraud_score(100, 'grocery') == 0.0
    assert calculate_simple_fraud_score(1500, 'grocery') == 0.3
    assert calculate_simple_fraud_score(500, 'casino') == 0.5
    assert calculate_simple_fraud_score(1500, 'casino') == 0.8

def test_data_processing():
    """Test data processing functions"""
    # Create sample data
    data = {
        'transaction_id': [f'tx_{i:03d}' for i in range(100)],
        'amount': np.random.uniform(10, 5000, 100),
        'merchant_id': np.random.choice(['amazon', 'walmart', 'casino'], 100),
        'timestamp': [datetime.now() for _ in range(100)]
    }
    
    df = pd.DataFrame(data)
    
    # Test data processing
    assert len(df) == 100
    assert all(col in df.columns for col in ['transaction_id', 'amount', 'merchant_id', 'timestamp'])
    
    # Test aggregations
    merchant_stats = df.groupby('merchant_id').agg({
        'amount': ['mean', 'count', 'sum']
    })
    
    assert len(merchant_stats) <= 3  # Max 3 unique merchants

def test_risk_level_assignment():
    """Test risk level assignment logic"""
    def assign_risk_level(fraud_score):
        if fraud_score >= 0.8:
            return 'CRITICAL'
        elif fraud_score >= 0.6:
            return 'HIGH'
        elif fraud_score >= 0.4:
            return 'MEDIUM'
        elif fraud_score >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    assert assign_risk_level(0.9) == 'CRITICAL'
    assert assign_risk_level(0.7) == 'HIGH'
    assert assign_risk_level(0.5) == 'MEDIUM'
    assert assign_risk_level(0.3) == 'LOW'
    assert assign_risk_level(0.1) == 'MINIMAL'

def test_decision_making():
    """Test transaction decision making logic"""
    def make_decision(fraud_score):
        if fraud_score >= 0.7:
            return 'DECLINED'
        elif fraud_score >= 0.4:
            return 'REVIEW'
        else:
            return 'APPROVED'
    
    assert make_decision(0.8) == 'DECLINED'
    assert make_decision(0.5) == 'REVIEW'
    assert make_decision(0.2) == 'APPROVED'

if __name__ == "__main__":
    pytest.main([__file__])