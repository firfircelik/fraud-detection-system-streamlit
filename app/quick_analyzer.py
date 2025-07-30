#!/usr/bin/env python3
"""
🚨 Streamlit Quick CSV Processor
Basit ve hızlı CSV fraud analizi aracı
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import tempfile
import os

# Page config
st.set_page_config(
    page_title="🚨 Quick CSV Fraud Analyzer",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Quick CSV Fraud Analyzer")
st.write("Basit ve hızlı CSV fraud analizi aracı")

# File upload
uploaded_file = st.file_uploader(
    "CSV dosyanızı yükleyin",
    type=['csv'],
    help="Transaction verilerinizi içeren CSV dosyasını seçin"
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dosya yüklendi: {len(df)} satır, {len(df.columns)} kolon")
        
        # Show preview
        with st.expander("📋 Veri Önizlemesi"):
            st.dataframe(df.head(10))
        
        # Basic analysis
        st.header("📊 Temel Analiz")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Toplam İşlem", len(df))
        
        with col2:
            if 'amount' in df.columns:
                avg_amount = df['amount'].mean()
                st.metric("💰 Ortalama Tutar", f"${avg_amount:.2f}")
            else:
                st.metric("💰 Ortalama Tutar", "N/A")
        
        with col3:
            if 'merchant_id' in df.columns:
                unique_merchants = df['merchant_id'].nunique()
                st.metric("🏪 Benzersiz Merchant", unique_merchants)
            else:
                st.metric("🏪 Benzersiz Merchant", "N/A")
        
        with col4:
            # Simulate fraud detection
            fraud_count = int(len(df) * 0.03)  # %3 fraud rate simulation
            st.metric("🚨 Fraud Tahmini", fraud_count)
        
        # Quick fraud simulation
        st.header("🔍 Hızlı Fraud Analizi")
        
        if st.button("🚀 Fraud Analizi Yap"):
            with st.spinner("Analiz yapılıyor..."):
                # Simulate processing
                df_copy = df.copy()
                
                # Add fraud score (simulation)
                np.random.seed(42)
                df_copy['fraud_score'] = np.random.beta(2, 10, len(df))
                
                # Add risk level
                df_copy['risk_level'] = pd.cut(
                    df_copy['fraud_score'],
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                )
                
                # Add decision
                df_copy['decision'] = df_copy['fraud_score'].apply(
                    lambda x: 'DECLINED' if x > 0.7 else ('REVIEW' if x > 0.4 else 'APPROVED')
                )
                
                st.success("✅ Analiz tamamlandı!")
                
                # Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Risk Dağılımı")
                    risk_counts = df_copy['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Seviyesi Dağılımı"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Karar Dağılımı")
                    decision_counts = df_copy['decision'].value_counts()
                    fig = px.bar(
                        x=decision_counts.index,
                        y=decision_counts.values,
                        title="İşlem Kararları",
                        color=decision_counts.values,
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Fraud score distribution
                st.subheader("📈 Fraud Skor Dağılımı")
                fig = px.histogram(
                    df_copy,
                    x='fraud_score',
                    title="Fraud Skorlarının Dağılımı",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # High risk transactions
                high_risk = df_copy[df_copy['fraud_score'] > 0.7]
                if len(high_risk) > 0:
                    st.subheader("🚨 Yüksek Riskli İşlemler")
                    st.dataframe(high_risk.head(20))
                
                # Download results
                st.subheader("📥 Sonuçları İndir")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_result = df_copy.to_csv(index=False)
                    st.download_button(
                        label="📄 CSV İndir",
                        data=csv_result,
                        file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Summary JSON
                    summary = {
                        'total_transactions': len(df_copy),
                        'fraud_rate': (df_copy['fraud_score'] > 0.7).mean(),
                        'risk_distribution': df_copy['risk_level'].value_counts().to_dict(),
                        'decision_distribution': df_copy['decision'].value_counts().to_dict(),
                        'analysis_date': datetime.now().isoformat()
                    }
                    
                    json_result = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📊 JSON Rapor İndir",
                        data=json_result,
                        file_name=f"fraud_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    except Exception as e:
        st.error(f"❌ Dosya işlenirken hata oluştu: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Başlamak için CSV dosyanızı yükleyin")
    
    # Sample data info
    with st.expander("📋 Beklenen CSV Formatı"):
        sample_data = {
            'transaction_id': ['tx_001', 'tx_002', 'tx_003'],
            'user_id': ['user_001', 'user_002', 'user_003'],
            'amount': [99.99, 1500.00, 25.50],
            'merchant_id': ['merchant_001', 'merchant_002', 'merchant_003'],
            'category': ['electronics', 'gambling', 'grocery'],
            'timestamp': ['2024-01-15T10:30:00Z', '2024-01-15T02:15:30Z', '2024-01-15T14:45:15Z']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        st.info("💡 Sistem otomatik olarak kolon isimlerini tanıyacaktır")

# Footer
st.divider()
st.markdown("🚨 **Streamlit Fraud Detection System** - Quick CSV Analyzer v1.0")
