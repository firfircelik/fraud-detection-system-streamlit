#!/usr/bin/env python3
"""
🚨 Fraud Detection System - Streamlit Cloud Version
Simplified standalone version for cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
import sys

# Page config
st.set_page_config(
    page_title="🚨 Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .fraud-high { color: #ff4444; font-weight: bold; }
    .fraud-medium { color: #ffaa00; font-weight: bold; }
    .fraud-low { color: #44ff44; font-weight: bold; }
    .status-approved { color: #44ff44; }
    .status-declined { color: #ff4444; }
    .status-review { color: #ffaa00; }
</style>
""", unsafe_allow_html=True)

# Main app
st.markdown('<h1 class="main-header">🚨 Advanced Fraud Detection System</h1>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Control Panel")
st.sidebar.success("✅ Streamlit Cloud: Online")
st.sidebar.info("🌐 Cloud Deployment Active")

# System status
st.sidebar.divider()
st.sidebar.write("📊 **System Status:**")
st.sidebar.metric("🔄 Status", "Active")
st.sidebar.metric("🌍 Platform", "Streamlit Cloud")
st.sidebar.metric("⚡ Response", "< 100ms")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Demo Dashboard", "🧪 Transaction Tester", "📄 CSV Upload"])

with tab1:
    st.header("📊 Fraud Detection Demo Dashboard")
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Sample Transactions",
            value="12,547",
            delta="+127 today"
        )
    
    with col2:
        st.metric(
            label="🚨 Fraud Detected",
            value="23",
            delta="+2 recent",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="📊 Fraud Rate",
            value="0.18%",
            delta="↓ 0.05%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="⚡ Processing",
            value="Real-time",
            delta="< 50ms"
        )
    
    st.divider()
    
    # Demo charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Fraud Trends (Demo)")
        
        # Generate demo data
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        fraud_counts = [2, 1, 4, 3, 0, 2, 1]
        
        df_demo = pd.DataFrame({
            'Date': dates,
            'Fraud_Count': fraud_counts
        })
        
        fig = px.line(df_demo, x='Date', y='Fraud_Count',
                     title="Fraud Detection Over Last 7 Days",
                     markers=True)
        fig.update_traces(line_color='#ff6b6b')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution (Demo)")
        
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Count': [8950, 2847, 687, 63]
        })
        
        fig = px.pie(risk_data, values='Count', names='Risk Level',
                    color_discrete_map={
                        'Low': '#44ff44',
                        'Medium': '#ffaa00', 
                        'High': '#ff8800',
                        'Critical': '#ff4444'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions demo
    st.subheader("📋 Recent Demo Transactions")
    
    demo_transactions = pd.DataFrame({
        'Transaction ID': ['tx_001', 'tx_002', 'tx_003', 'tx_004', 'tx_005'],
        'Amount': [125.50, 2499.99, 19.99, 999.00, 75.25],
        'Merchant': ['Amazon', 'Electronics Store', 'Coffee Shop', 'Crypto Exchange', 'Gas Station'],
        'Risk Score': [0.12, 0.85, 0.05, 0.92, 0.08],
        'Status': ['Approved', 'Review', 'Approved', 'Declined', 'Approved']
    })
    
    # Style the dataframe
    def color_status(val):
        if val == 'Approved':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Declined':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #fff3cd; color: #856404'
    
    def color_risk(val):
        if val > 0.7:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        elif val > 0.3:
            return 'background-color: #fff3cd; color: #856404; font-weight: bold'
        else:
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
    
    styled_df = demo_transactions.style.apply(lambda x: [color_status(val) if x.name == 'Status' else '' for val in x], axis=1) \
                                           .apply(lambda x: [color_risk(val) if x.name == 'Risk Score' else '' for val in x], axis=1) \
                                           .format({'Amount': '${:.2f}', 'Risk Score': '{:.2f}'})
    
    st.dataframe(styled_df, use_container_width=True)

with tab2:
    st.header("🧪 Transaction Fraud Tester")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Transaction Details")
        
        account_id = st.text_input("Account ID", value="acc_test_user")
        merchant_id = st.selectbox("Merchant", 
                                  ["amazon", "walmart", "grocery_store", "electronics", "gambling_site", "crypto_exchange"])
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=50000.0, value=100.0)
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP"])
        
        if st.button("🚀 Analyze Transaction", type="primary"):
            # Simulate fraud analysis
            risk_score = 0.0
            
            # Simple rule-based fraud detection
            if merchant_id in ["gambling_site", "crypto_exchange"]:
                risk_score += 0.4
            
            if amount > 1000:
                risk_score += 0.3
            elif amount > 5000:
                risk_score += 0.5
            
            if "suspicious" in account_id.lower():
                risk_score += 0.6
            
            # Add some randomness
            risk_score += np.random.uniform(0, 0.2)
            risk_score = min(risk_score, 1.0)
            
            # Determine status
            if risk_score > 0.7:
                status = "🚨 DECLINED"
                risk_level = "HIGH"
                color = "red"
            elif risk_score > 0.4:
                status = "⚠️ REVIEW"
                risk_level = "MEDIUM"
                color = "orange"
            else:
                status = "✅ APPROVED"
                risk_level = "LOW"
                color = "green"
            
            st.success("✅ Transaction analyzed!")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("🎯 Risk Score", f"{risk_score:.1%}")
            
            with col_b:
                st.metric("📊 Risk Level", risk_level)
            
            with col_c:
                st.metric("⚖️ Decision", status)
            
            # Show analysis details
            st.json({
                "transaction_id": f"tx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "account_id": account_id,
                "merchant_id": merchant_id,
                "amount": amount,
                "currency": currency,
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "decision": status,
                "timestamp": datetime.now().isoformat()
            })
    
    with col2:
        st.subheader("🎯 Quick Test Scenarios")
        
        scenarios = [
            {"name": "✅ Normal Shopping", "merchant": "amazon", "amount": 99.99},
            {"name": "⚠️ High Amount", "merchant": "electronics", "amount": 2500.0},
            {"name": "🚨 Gambling", "merchant": "gambling_site", "amount": 500.0},
            {"name": "💰 Crypto", "merchant": "crypto_exchange", "amount": 1500.0}
        ]
        
        for scenario in scenarios:
            with st.expander(scenario["name"]):
                st.write(f"**Merchant:** {scenario['merchant']}")
                st.write(f"**Amount:** ${scenario['amount']:.2f}")
                
                if st.button(f"Test {scenario['name']}", key=scenario['name']):
                    st.info(f"Testing {scenario['name']}...")

with tab3:
    st.header("📄 CSV Transaction Processor")
    
    st.info("🔧 **Coming Soon**: Full CSV processing capabilities")
    st.write("Upload your transaction data for batch fraud analysis.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded: {len(df)} transactions")
        
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        if st.button("🚀 Process for Fraud Detection"):
            with st.spinner("Processing transactions..."):
                time.sleep(2)  # Simulate processing
                
                # Simple demo processing
                df['fraud_score'] = np.random.uniform(0, 1, len(df))
                df['risk_level'] = pd.cut(df['fraud_score'], 
                                        bins=[0, 0.3, 0.6, 0.8, 1.0],
                                        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
                
                st.success("✅ Processing complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fraud_count = len(df[df['fraud_score'] > 0.7])
                    st.metric("🚨 High Risk Transactions", fraud_count)
                
                with col2:
                    avg_risk = df['fraud_score'].mean()
                    st.metric("📊 Average Risk Score", f"{avg_risk:.1%}")
                
                st.subheader("📈 Risk Distribution")
                fig = px.histogram(df, x='fraud_score', nbins=20,
                                 title="Risk Score Distribution")
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🌐 **Streamlit Cloud Deployment**")

with col2:
    st.success("⚡ **Real-time Processing**")

with col3:
    st.warning("🔧 **Demo Version**")

st.caption("🚨 Advanced Fraud Detection System - Cloud Demo Version")
