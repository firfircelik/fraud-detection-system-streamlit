#!/usr/bin/env python3
"""
Unit tests for CSV Fraud Processor
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime
import sys
sys.path.append('..')
from csv_fraud_processor import CSVFraudProcessor

class TestCSVFraudProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = CSVFraudProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'transaction_id': ['tx_001', 'tx_002', 'tx_003'],
            'user_id': ['user_001', 'user_002', 'user_003'],
            'amount': [100.0, 5000.0, 10.0],
            'merchant_id': ['merchant_001', 'gambling_site', 'grocery_store'],
            'category': ['electronics', 'gambling', 'grocery'],
            'timestamp': [datetime.now().isoformat()] * 3,
            'currency': ['USD', 'USD', 'USD']
        })
    
    def test_standardize_columns(self):
        """Test column standardization"""
        # Test data with different column names
        test_df = pd.DataFrame({
            'TransactionID': ['tx_001'],
            'Amount': [100.0],
            'UserID': ['user_001']
        })
        
        standardized = self.processor.standardize_columns(test_df)
        
        self.assertIn('transaction_id', standardized.columns)
        self.assertIn('amount', standardized.columns)
        self.assertIn('user_id', standardized.columns)
    
    def test_calculate_fraud_score(self):
        """Test fraud score calculation"""
        # High amount transaction
        high_amount_row = pd.Series({
            'amount': 6000.0,
            'merchant_id': 'normal_merchant',
            'category': 'electronics',
            'currency': 'USD'
        })
        
        score, risk_level, risk_factors = self.processor.calculate_fraud_score(high_amount_row)
        
        self.assertGreater(score, 0.4)  # Should be high risk
        self.assertIn('very_high_amount', risk_factors)
    
    def test_suspicious_merchant(self):
        """Test suspicious merchant detection"""
        suspicious_row = pd.Series({
            'amount': 100.0,
            'merchant_id': 'gambling_site',
            'category': 'gambling',
            'currency': 'USD'
        })
        
        score, risk_level, risk_factors = self.processor.calculate_fraud_score(suspicious_row)
        
        self.assertIn('suspicious_merchant', risk_factors)
        self.assertIn('suspicious_category', risk_factors)
    
    def test_process_batch(self):
        """Test batch processing"""
        processed = self.processor.process_batch(self.sample_data)
        
        # Check required columns are added
        self.assertIn('fraud_score', processed.columns)
        self.assertIn('risk_level', processed.columns)
        self.assertIn('risk_factors', processed.columns)
        self.assertIn('decision', processed.columns)
        
        # Check all scores are between 0 and 1
        self.assertTrue(all(0 <= score <= 1 for score in processed['fraud_score']))
    
    def test_generate_summary_report(self):
        """Test summary report generation"""
        processed = self.processor.process_batch(self.sample_data)
        summary = self.processor.generate_summary_report(processed)
        
        # Check required fields
        self.assertIn('total_transactions', summary)
        self.assertIn('decisions', summary)
        self.assertIn('risk_levels', summary)
        self.assertIn('fraud_score_stats', summary)
        
        self.assertEqual(summary['total_transactions'], len(self.sample_data))
    
    def test_save_results(self):
        """Test results saving"""
        processed = self.processor.process_batch(self.sample_data)
        summary = self.processor.generate_summary_report(processed)
        
        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            original_results_dir = self.processor.results_dir
            self.processor.results_dir = temp_dir
            
            csv_path, json_path = self.processor.save_results(processed, summary, 'test')
            
            # Check files exist
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))
            
            # Restore original directory
            self.processor.results_dir = original_results_dir

if __name__ == '__main__':
    unittest.main()
