#!/usr/bin/env python3
"""
Quick fix script for Streamlit CSV processing
"""

import streamlit as st
import pandas as pd
from csv_fraud_processor import CSVFraudProcessor
from datetime import datetime
import tempfile
import shutil
import os

def main():
    st.title("🔧 CSV Processing Debug Tool")
    
    # Test CSV processor
    st.header("1. Test CSV Processor")
    if st.button("Test Processor"):
        try:
            processor = CSVFraudProcessor()
            st.success("✅ CSV Processor loaded successfully")
            
            # Test with sample data
            test_data = {
                'transaction_id': ['tx_001', 'tx_002'],
                'user_id': ['user_001', 'user_002'],
                'amount': [99.99, 5500.00],
                'merchant_id': ['merchant_normal', 'merchant_gambling'],
                'category': ['electronics', 'gambling'],
                'currency': ['USD', 'USD']
            }
            
            df = pd.DataFrame(test_data)
            df_processed = processor.process_batch(df)
            summary = processor.generate_summary_report(df_processed)
            
            st.success("✅ Processing test completed")
            st.json(summary)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    # File upload test
    st.header("2. File Upload Test")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        st.write(f"File size: {uploaded_file.size} bytes")
        st.write(f"File type: {uploaded_file.type}")
        
        if st.button("Process File"):
            try:
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)
                    tmp_path = tmp_file.name
                
                # Read and process
                df = pd.read_csv(tmp_path)
                st.success(f"✅ Read {len(df)} rows")
                st.dataframe(df.head())
                
                # Process
                processor = CSVFraudProcessor()
                df_processed = processor.process_batch(df)
                summary = processor.generate_summary_report(df_processed)
                
                st.success("✅ Processing completed")
                st.json(summary)
                
                # Clean up
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Session state debug
    st.header("3. Session State Debug")
    st.write("Session State:", dict(st.session_state))

if __name__ == "__main__":
    main()