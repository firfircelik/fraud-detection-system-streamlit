#!/usr/bin/env python3
"""
Fixed CSV Processor for Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from csv_fraud_processor import CSVFraudProcessor
from datetime import datetime
import tempfile
import shutil
import os

def show_csv_processor_fixed():
    """Fixed CSV processor with better state management"""
    
    st.header("📄 CSV Fraud Detection Processor")
    st.write("Upload and process CSV files for batch fraud detection")
    
    # Initialize processor
    try:
        processor = CSVFraudProcessor()
        st.success("✅ CSV Processor ready!")
    except Exception as e:
        st.error(f"❌ CSV Processor error: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your transaction data for fraud analysis",
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process immediately when file is uploaded
        try:
            # Read file
            df = pd.read_csv(uploaded_file)
            st.info(f"📊 Found {len(df)} transactions with {len(df.columns)} columns")
            
            # Show preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head())
            
            # Processing options
            col1, col2 = st.columns(2)
            
            with col1:
                sample_size = st.number_input(
                    "Sample Size (0 = all)",
                    min_value=0,
                    max_value=len(df),
                    value=min(1000, len(df)),
                    key="sample_size_input"
                )
            
            with col2:
                process_mode = st.selectbox(
                    "Processing Mode",
                    ["Quick Analysis", "Full Processing"],
                    key="process_mode_select"
                )
            
            # Process button
            if st.button("🚀 Analyze for Fraud", type="primary", key="analyze_button"):
                
                # Create containers for results
                results_container = st.container()
                
                with st.spinner("🔍 Analyzing transactions..."):
                    
                    # Prepare data
                    if sample_size > 0 and sample_size < len(df):
                        df_sample = df.sample(n=sample_size, random_state=42)
                    else:
                        df_sample = df
                    
                    # Process
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("🔍 Running fraud detection...")
                    progress.progress(25)
                    
                    df_processed = processor.process_batch(df_sample)
                    
                    status.text("📊 Generating summary...")
                    progress.progress(50)
                    
                    summary = processor.generate_summary_report(df_processed)
                    
                    status.text("💾 Saving results...")
                    progress.progress(75)
                    
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_prefix = f"uploaded_{uploaded_file.name.replace('.csv', '')}_{timestamp}"
                    csv_path, json_path = processor.save_results(df_processed, summary, filename_prefix)
                    
                    progress.progress(100)
                    status.text("✅ Analysis complete!")
                    
                    # Show results immediately in the container
                    with results_container:
                        show_results_immediately(summary, df_processed, csv_path, json_path, uploaded_file.name)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    else:
        # Show expected format
        st.info("👆 Upload a CSV file to start fraud analysis")
        
        st.subheader("📋 Expected CSV Format")
        sample_data = {
            'transaction_id': ['tx_001', 'tx_002', 'tx_003'],
            'user_id': ['user_001', 'user_002', 'user_003'],
            'amount': [99.99, 1500.00, 25.50],
            'merchant_id': ['merchant_001', 'merchant_002', 'merchant_003'],
            'category': ['electronics', 'gambling', 'grocery']
        }
        
        st.dataframe(pd.DataFrame(sample_data))

def show_results_immediately(summary, df_processed, csv_path, json_path, filename):
    """Show results immediately without session state"""
    
    st.header("📊 Fraud Analysis Results")
    st.success(f"🎉 Analysis completed for: {filename}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", f"{summary['total_transactions']:,}")
    
    with col2:
        approved = summary['decisions']['approved']
        st.metric("Approved", f"{approved:,}")
    
    with col3:
        declined = summary['decisions']['declined']
        st.metric("Declined", f"{declined:,}")
    
    with col4:
        fraud_rate = summary['fraud_rate'] * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Decisions")
        decisions = summary['decisions']
        
        decision_data = pd.DataFrame([
            {'Decision': 'Approved', 'Count': decisions['approved']},
            {'Decision': 'Review', 'Count': decisions['review']},
            {'Decision': 'Declined', 'Count': decisions['declined']}
        ])
        
        fig = px.pie(decision_data, values='Count', names='Decision',
                    color_discrete_map={
                        'Approved': '#00ff00',
                        'Review': '#ffaa00',
                        'Declined': '#ff0000'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Levels")
        risk_levels = summary['risk_levels']
        
        risk_data = pd.DataFrame([
            {'Risk Level': level.title(), 'Count': count}
            for level, count in risk_levels.items()
            if count > 0
        ])
        
        if not risk_data.empty:
            fig = px.bar(risk_data, x='Risk Level', y='Count',
                        color='Count', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    if summary['top_risk_factors']:
        st.subheader("⚠️ Top Risk Factors")
        risk_factors = summary['top_risk_factors']
        risk_df = pd.DataFrame(list(risk_factors.items()), 
                             columns=['Risk Factor', 'Count'])
        
        fig = px.bar(risk_df, x='Count', y='Risk Factor', orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample results
    st.subheader("📋 Sample Results")
    display_cols = ['transaction_id', 'amount', 'merchant_id', 'fraud_score', 'risk_level', 'decision']
    available_cols = [col for col in display_cols if col in df_processed.columns]
    st.dataframe(df_processed[available_cols].head(10))
    
    # Download buttons
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'rb') as f:
                st.download_button(
                    "📥 Download CSV",
                    data=f.read(),
                    file_name=os.path.basename(csv_path),
                    mime='text/csv'
                )
    
    with col2:
        if json_path and os.path.exists(json_path):
            with open(json_path, 'rb') as f:
                st.download_button(
                    "📥 Download JSON",
                    data=f.read(),
                    file_name=os.path.basename(json_path),
                    mime='application/json'
                )

# Test the fixed processor
if __name__ == "__main__":
    st.set_page_config(page_title="Fixed CSV Processor", layout="wide")
    show_csv_processor_fixed()