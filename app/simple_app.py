#!/usr/bin/env python3
"""
Simple CSV Fraud Detection App
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from fraud_processor import CSVFraudProcessor
from datetime import datetime
import tempfile
import shutil
import os

st.set_page_config(
    page_title="🚨 CSV Fraud Detector",
    page_icon="🚨",
    layout="wide"
)

def main():
    st.title("🚨 CSV Fraud Detection System")
    st.write("Upload your CSV file and get instant fraud analysis!")
    
    # Initialize processor
    try:
        processor = CSVFraudProcessor()
        st.success("✅ Fraud detection system ready!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return
    
    # File upload
    st.header("📤 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your transaction data for fraud analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} transactions")
            
            # Show preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head(10))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**File Info:**")
                    st.write(f"- Rows: {len(df):,}")
                    st.write(f"- Columns: {len(df.columns)}")
                    st.write(f"- Size: {uploaded_file.size:,} bytes")
                
                with col2:
                    st.write("**Columns:**")
                    for col in df.columns:
                        st.write(f"- {col}")
            
            # Processing options
            st.header("⚙️ Processing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sample_size = st.number_input(
                    "Sample Size (0 = process all)",
                    min_value=0,
                    max_value=len(df),
                    value=min(1000, len(df)),
                    step=100
                )
            
            with col2:
                process_mode = st.selectbox(
                    "Processing Mode",
                    ["Quick Analysis", "Full Processing"],
                    help="Quick = faster, Full = complete analysis"
                )
            
            # Process button
            if st.button("🚀 Analyze for Fraud", type="primary", use_container_width=True):
                
                with st.spinner("🔍 Analyzing transactions for fraud..."):
                    
                    # Prepare data
                    if sample_size > 0 and sample_size < len(df):
                        df_sample = df.sample(n=sample_size, random_state=42)
                        st.info(f"Processing {len(df_sample)} sampled transactions...")
                    else:
                        df_sample = df
                        st.info(f"Processing all {len(df_sample)} transactions...")
                    
                    # Process
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Running fraud detection...")
                    progress_bar.progress(25)
                    
                    df_processed = processor.process_batch(df_sample)
                    
                    status_text.text("📊 Generating summary...")
                    progress_bar.progress(50)
                    
                    summary = processor.generate_summary_report(df_processed)
                    
                    status_text.text("💾 Saving results...")
                    progress_bar.progress(75)
                    
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_prefix = f"uploaded_{uploaded_file.name.replace('.csv', '')}_{timestamp}"
                    csv_path, json_path = processor.save_results(df_processed, summary, filename_prefix)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    st.success("🎉 Fraud analysis completed successfully!")
                
                # Show results
                show_results(summary, df_processed, csv_path, json_path)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Show sample format
        st.header("📋 Expected CSV Format")
        st.write("Your CSV should contain these columns (or similar):")
        
        sample_data = {
            'transaction_id': ['tx_001', 'tx_002', 'tx_003'],
            'user_id': ['user_001', 'user_002', 'user_003'],
            'amount': [99.99, 1500.00, 25.50],
            'merchant_id': ['merchant_001', 'merchant_002', 'merchant_003'],
            'category': ['electronics', 'gambling', 'grocery'],
            'timestamp': ['2024-01-15T10:30:00Z', '2024-01-15T02:15:30Z', '2024-01-15T14:45:15Z']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.info("💡 The system automatically maps common column variations")

def show_results(summary, df_processed, csv_path, json_path):
    """Show analysis results"""
    
    st.header("📊 Fraud Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{summary['total_transactions']:,}"
        )
    
    with col2:
        approved = summary['decisions']['approved']
        st.metric(
            "Approved",
            f"{approved:,}",
            delta=f"{approved/summary['total_transactions']*100:.1f}%"
        )
    
    with col3:
        declined = summary['decisions']['declined']
        st.metric(
            "Declined",
            f"{declined:,}",
            delta=f"{declined/summary['total_transactions']*100:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        fraud_rate = summary['fraud_rate'] * 100
        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%",
            delta="High" if fraud_rate > 5 else "Normal"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Decision Breakdown")
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
        st.subheader("🎯 Risk Level Distribution")
        risk_levels = summary['risk_levels']
        
        risk_data = pd.DataFrame([
            {'Risk Level': level.title(), 'Count': count}
            for level, count in risk_levels.items()
            if count > 0
        ])
        
        if not risk_data.empty:
            fig = px.bar(risk_data, x='Risk Level', y='Count',
                        color='Count',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    if summary['top_risk_factors']:
        st.subheader("⚠️ Top Risk Factors")
        
        risk_factors = summary['top_risk_factors']
        risk_df = pd.DataFrame(list(risk_factors.items()), 
                             columns=['Risk Factor', 'Count'])
        risk_df = risk_df.head(10)
        
        fig = px.bar(risk_df, x='Count', y='Risk Factor',
                    orientation='h',
                    title="Most Common Risk Factors")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample results
    st.subheader("📋 Sample Processed Transactions")
    
    # Show high-risk transactions
    high_risk = df_processed[df_processed['risk_level'].isin(['HIGH', 'CRITICAL'])]
    if not high_risk.empty:
        st.write("**🚨 High Risk Transactions:**")
        st.dataframe(high_risk[['transaction_id', 'amount', 'merchant_id', 'fraud_score', 'risk_level', 'decision']].head(10))
    
    # Show sample of all results
    st.write("**📊 All Results (Sample):**")
    display_cols = ['transaction_id', 'user_id', 'amount', 'merchant_id', 'fraud_score', 'risk_level', 'decision']
    available_cols = [col for col in display_cols if col in df_processed.columns]
    st.dataframe(df_processed[available_cols].head(20))
    
    # Download options
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(csv_path):
            with open(csv_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed CSV",
                    data=f.read(),
                    file_name=os.path.basename(csv_path),
                    mime='text/csv',
                    use_container_width=True
                )
    
    with col2:
        if os.path.exists(json_path):
            with open(json_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Summary JSON",
                    data=f.read(),
                    file_name=os.path.basename(json_path),
                    mime='application/json',
                    use_container_width=True
                )

if __name__ == "__main__":
    main()