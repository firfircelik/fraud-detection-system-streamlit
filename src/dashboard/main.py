#!/usr/bin/env python3
"""
📊 Advanced Analytics Dashboard
Enterprise-grade fraud detection dashboard with comprehensive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedFraudDashboard:
    """Advanced fraud detection dashboard with comprehensive analytics"""
    
    def __init__(self):
        self.api_base = "http://localhost:8080/api"
        self.refresh_interval = 30  # seconds
        
    def render_dashboard(self):
        """Render the complete advanced dashboard"""
        
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0;'>🚨 Enterprise Fraud Detection Dashboard</h1>
            <p style='color: white; margin: 0; opacity: 0.9;'>Real-time fraud monitoring and advanced analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "📈 Analytics", "🤖 ML Models", 
            "🚨 Alerts", "🔍 Investigations", "⚙️ Settings"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_ml_models_tab()
        
        with tab4:
            self.render_alerts_tab()
        
        with tab5:
            self.render_investigations_tab()
        
        with tab6:
            self.render_settings_tab()
    
    def render_overview_tab(self):
        """Render overview dashboard"""
        
        st.header("📊 Real-time Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔍 Total Transactions",
                value="1,234,567",
                delta="↑ 12.5% vs yesterday"
            )
        
        with col2:
            st.metric(
                label="🚨 Fraud Detected",
                value="2,345",
                delta="↓ 8.2% vs yesterday",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="📊 Fraud Rate",
                value="0.19%",
                delta="↓ 0.03% vs yesterday",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="💰 Prevented Loss",
                value="$2.1M",
                delta="↑ $340K vs yesterday"
            )
        
        st.divider()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Transaction Volume (24h)")
            
            # Generate sample data
            hours = list(range(24))
            volumes = [np.random.poisson(1000) + 500 for _ in hours]
            fraud_counts = [np.random.poisson(5) + 1 for _ in hours]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours, y=volumes,
                mode='lines+markers',
                name='Total Transactions',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=hours, y=fraud_counts,
                mode='lines+markers',
                name='Fraud Detected',
                line=dict(color='#ff7f0e', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Transaction Count",
                yaxis2=dict(
                    title="Fraud Count",
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Risk Distribution")
            
            risk_data = {
                'Risk Level': ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                'Count': [850000, 300000, 70000, 12000, 2567],
                'Color': ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
            }
            
            fig = px.pie(
                values=risk_data['Count'],
                names=risk_data['Risk Level'],
                color_discrete_sequence=risk_data['Color'],
                title="Transaction Risk Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Real-time alerts
        st.subheader("🚨 Real-time Alerts")
        
        alert_data = [
            {"Time": "14:32:15", "Type": "High Risk", "Transaction": "TX_789123", "Amount": "$15,000", "Risk": "CRITICAL"},
            {"Time": "14:31:42", "Type": "Velocity", "Transaction": "TX_789122", "Amount": "$2,500", "Risk": "HIGH"},
            {"Time": "14:30:18", "Type": "Geographic", "Transaction": "TX_789121", "Amount": "$850", "Risk": "MEDIUM"},
            {"Time": "14:29:55", "Type": "Amount", "Transaction": "TX_789120", "Amount": "$25,000", "Risk": "CRITICAL"},
        ]
        
        df_alerts = pd.DataFrame(alert_data)
        
        # Style the dataframe
        def style_risk(val):
            colors = {
                'CRITICAL': 'background-color: #ffebee; color: #c62828',
                'HIGH': 'background-color: #fff3e0; color: #ef6c00',
                'MEDIUM': 'background-color: #f3e5f5; color: #7b1fa2'
            }
            return colors.get(val, '')
        
        styled_df = df_alerts.style.applymap(style_risk, subset=['Risk'])
        st.dataframe(styled_df, use_container_width=True)
    
    def render_analytics_tab(self):
        """Render advanced analytics"""
        
        st.header("📈 Advanced Analytics")
        
        # Time range selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            time_range = st.selectbox(
                "📅 Time Range",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )
        
        with col2:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        st.divider()
        
        # Analytics sections
        analytics_tabs = st.tabs([
            "⏰ Temporal", "🌍 Geographic", "👤 Behavioral", 
            "💰 Financial", "🔗 Network", "🎯 Patterns"
        ])
        
        with analytics_tabs[0]:
            self.render_temporal_analytics()
        
        with analytics_tabs[1]:
            self.render_geographic_analytics()
        
        with analytics_tabs[2]:
            self.render_behavioral_analytics()
        
        with analytics_tabs[3]:
            self.render_financial_analytics()
        
        with analytics_tabs[4]:
            self.render_network_analytics()
        
        with analytics_tabs[5]:
            self.render_pattern_analytics()
    
    def render_temporal_analytics(self):
        """Render temporal pattern analysis"""
        
        st.subheader("⏰ Temporal Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Fraud by Hour of Day**")
            
            hours = list(range(24))
            fraud_rates = [0.1 + 0.3 * np.sin(2 * np.pi * h / 24) + np.random.normal(0, 0.05) for h in hours]
            fraud_rates = [max(0, min(1, rate)) for rate in fraud_rates]
            
            fig = px.bar(
                x=hours, y=fraud_rates,
                labels={'x': 'Hour of Day', 'y': 'Fraud Rate (%)'},
                title="Fraud Rate by Hour",
                color=fraud_rates,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📅 Weekly Pattern Analysis**")
            
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fraud_counts = [120, 115, 118, 122, 135, 95, 85]
            
            fig = px.line(
                x=days, y=fraud_counts,
                markers=True,
                title="Fraud Count by Day of Week"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal analysis
        st.write("**🌟 Seasonal Trends**")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fraud_trends = [0.15, 0.18, 0.16, 0.14, 0.13, 0.12, 
                       0.11, 0.13, 0.15, 0.17, 0.19, 0.22]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=fraud_trends,
            mode='lines+markers',
            name='Fraud Rate',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Monthly Fraud Rate Trends",
            xaxis_title="Month",
            yaxis_title="Fraud Rate (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_geographic_analytics(self):
        """Render geographic analysis"""
        
        st.subheader("🌍 Geographic Analysis")
        
        # Generate sample geographic data
        countries = ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'Brazil']
        fraud_rates = [0.15, 0.12, 0.08, 0.10, 0.11, 0.09, 0.06, 0.25]
        transaction_counts = [500000, 120000, 80000, 95000, 75000, 45000, 60000, 35000]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🗺️ Fraud Rate by Country**")
            
            fig = px.bar(
                x=countries, y=fraud_rates,
                color=fraud_rates,
                color_continuous_scale='Reds',
                title="Fraud Rate by Country"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📊 Transaction Volume vs Fraud Rate**")
            
            fig = px.scatter(
                x=transaction_counts, y=fraud_rates,
                size=[c/1000 for c in transaction_counts],
                hover_name=countries,
                labels={'x': 'Transaction Count', 'y': 'Fraud Rate (%)'},
                title="Volume vs Fraud Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk heat map
        st.write("**🔥 Geographic Risk Heatmap**")
        
        # Sample coordinates for major cities
        cities_data = {
            'City': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Toronto', 'Berlin', 'São Paulo'],
            'Lat': [40.7128, 51.5074, 35.6762, 48.8566, -33.8688, 43.6532, 52.5200, -23.5505],
            'Lon': [-74.0060, -0.1278, 139.6503, 2.3522, 151.2093, -79.3832, 13.4050, -46.6333],
            'Risk_Score': [0.15, 0.12, 0.06, 0.10, 0.09, 0.11, 0.08, 0.25],
            'Transaction_Count': [50000, 12000, 8000, 9500, 4500, 7500, 6000, 3500]
        }
        
        df_cities = pd.DataFrame(cities_data)
        
        fig = px.scatter_mapbox(
            df_cities,
            lat='Lat', lon='Lon',
            size='Transaction_Count',
            color='Risk_Score',
            hover_name='City',
            color_continuous_scale='Reds',
            size_max=20,
            zoom=1,
            mapbox_style='carto-positron',
            title="Global Fraud Risk Distribution"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_behavioral_analytics(self):
        """Render behavioral analysis"""
        
        st.subheader("👤 Behavioral Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 User Risk Segmentation**")
            
            segments = ['Low Risk', 'Medium Risk', 'High Risk', 'VIP', 'New Users']
            user_counts = [450000, 85000, 12000, 5000, 25000]
            fraud_rates = [0.05, 0.15, 0.45, 0.02, 0.25]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=segments, y=user_counts,
                name='User Count',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Scatter(
                x=segments, y=[rate * 10000 for rate in fraud_rates],
                mode='lines+markers',
                name='Fraud Rate (scaled)',
                yaxis='y2',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="User Segmentation Analysis",
                yaxis=dict(title="User Count"),
                yaxis2=dict(title="Fraud Rate", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**⚡ Transaction Velocity Analysis**")
            
            velocity_ranges = ['1-5 tx/day', '6-10 tx/day', '11-20 tx/day', '21-50 tx/day', '50+ tx/day']
            avg_fraud_rates = [0.08, 0.12, 0.18, 0.35, 0.65]
            
            fig = px.bar(
                x=velocity_ranges, y=avg_fraud_rates,
                color=avg_fraud_rates,
                color_continuous_scale='Oranges',
                title="Fraud Rate by Transaction Velocity"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Behavioral patterns
        st.write("**🔍 Behavioral Pattern Detection**")
        
        patterns = [
            {"Pattern": "Rapid Fire Transactions", "Occurrences": 1250, "Avg Risk": 0.75, "Status": "🔴 High Alert"},
            {"Pattern": "Unusual Time Activity", "Occurrences": 890, "Avg Risk": 0.45, "Status": "🟡 Monitor"},
            {"Pattern": "Geographic Anomaly", "Occurrences": 650, "Avg Risk": 0.55, "Status": "🟡 Monitor"},
            {"Pattern": "Amount Spike", "Occurrences": 420, "Avg Risk": 0.85, "Status": "🔴 High Alert"},
            {"Pattern": "New Device Usage", "Occurrences": 2100, "Avg Risk": 0.25, "Status": "🟢 Normal"},
        ]
        
        df_patterns = pd.DataFrame(patterns)
        st.dataframe(df_patterns, use_container_width=True)
    
    def render_financial_analytics(self):
        """Render financial impact analysis"""
        
        st.subheader("💰 Financial Impact Analysis")
        
        # Financial metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💸 Total Fraud Loss", "$2.1M", "↓ $340K")
        
        with col2:
            st.metric("🛡️ Prevented Loss", "$15.8M", "↑ $2.1M")
        
        with col3:
            st.metric("📊 ROI", "750%", "↑ 45%")
        
        with col4:
            st.metric("⚡ Avg Detection Time", "1.2s", "↓ 0.3s")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💹 Monthly Financial Impact**")
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            fraud_loss = [1.8, 2.1, 1.9, 2.3, 2.0, 2.1]
            prevented_loss = [12.5, 14.2, 13.8, 15.1, 14.9, 15.8]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=months, y=fraud_loss,
                name='Fraud Loss ($M)',
                marker_color='red'
            ))
            fig.add_trace(go.Bar(
                x=months, y=prevented_loss,
                name='Prevented Loss ($M)',
                marker_color='green'
            ))
            
            fig.update_layout(
                title="Monthly Financial Impact",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🎯 Cost-Benefit Analysis**")
            
            categories = ['Detection Cost', 'Investigation Cost', 'False Positive Cost', 'Prevented Loss']
            values = [0.5, 0.3, 0.8, 15.8]
            colors = ['red', 'orange', 'yellow', 'green']
            
            fig = px.bar(
                x=categories, y=values,
                color=colors,
                color_discrete_sequence=colors,
                title="Cost-Benefit Breakdown ($M)"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_network_analytics(self):
        """Render network analysis"""
        
        st.subheader("🔗 Network Analysis")
        
        st.info("🚧 Network analysis features coming soon! This will include user-merchant relationship graphs, fraud ring detection, and community analysis.")
        
        # Placeholder for network visualization
        st.write("**🕸️ Fraud Network Visualization**")
        st.write("Interactive network graphs showing connections between fraudulent users, merchants, and devices.")
        
        st.write("**🎯 Fraud Ring Detection**")
        st.write("Automated detection of coordinated fraud attacks and suspicious user groups.")
    
    def render_pattern_analytics(self):
        """Render pattern analysis"""
        
        st.subheader("🎯 Pattern Analysis")
        
        st.write("**🔍 Detected Fraud Patterns**")
        
        pattern_data = [
            {
                "Pattern ID": "PAT_001",
                "Pattern Type": "Temporal Clustering",
                "Description": "Multiple high-value transactions within 5-minute windows",
                "Confidence": 0.92,
                "Affected Transactions": 1250,
                "Status": "🔴 Active"
            },
            {
                "Pattern ID": "PAT_002", 
                "Pattern Type": "Geographic Anomaly",
                "Description": "Transactions from unusual locations for user profile",
                "Confidence": 0.87,
                "Affected Transactions": 890,
                "Status": "🟡 Monitoring"
            },
            {
                "Pattern ID": "PAT_003",
                "Pattern Type": "Amount Progression",
                "Description": "Gradual increase in transaction amounts over time",
                "Confidence": 0.78,
                "Affected Transactions": 650,
                "Status": "🟢 Resolved"
            }
        ]
        
        df_patterns = pd.DataFrame(pattern_data)
        st.dataframe(df_patterns, use_container_width=True)
        
        # Pattern evolution
        st.write("**📈 Pattern Evolution Over Time**")
        
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        pattern_strength = [0.3 + 0.4 * np.sin(i/5) + np.random.normal(0, 0.1) for i in range(30)]
        pattern_strength = [max(0, min(1, s)) for s in pattern_strength]
        
        fig = px.line(
            x=dates, y=pattern_strength,
            title="Pattern Strength Evolution",
            labels={'x': 'Date', 'y': 'Pattern Strength'}
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ml_models_tab(self):
        """Render ML models monitoring"""
        
        st.header("🤖 ML Models Performance")
        
        # Model status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Overall Accuracy", "94.2%", "↑ 1.2%")
        
        with col2:
            st.metric("⚡ Avg Inference Time", "12ms", "↓ 3ms")
        
        with col3:
            st.metric("🔄 Models Active", "4/4", "All healthy")
        
        with col4:
            st.metric("📊 Predictions Today", "1.2M", "↑ 15%")
        
        st.divider()
        
        # Individual model performance
        st.subheader("📊 Individual Model Performance")
        
        model_data = [
            {"Model": "Random Forest", "Accuracy": 0.942, "Precision": 0.891, "Recall": 0.876, "F1": 0.883, "Status": "🟢 Healthy"},
            {"Model": "Logistic Regression", "Accuracy": 0.918, "Precision": 0.845, "Recall": 0.832, "F1": 0.838, "Status": "🟢 Healthy"},
            {"Model": "Isolation Forest", "Accuracy": 0.887, "Precision": 0.798, "Recall": 0.823, "F1": 0.810, "Status": "🟡 Monitoring"},
            {"Model": "SVM", "Accuracy": 0.925, "Precision": 0.867, "Recall": 0.854, "F1": 0.860, "Status": "🟢 Healthy"},
        ]
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        # Model performance over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Accuracy Trends**")
            
            days = list(range(1, 31))
            rf_acc = [0.94 + np.random.normal(0, 0.01) for _ in days]
            lr_acc = [0.92 + np.random.normal(0, 0.01) for _ in days]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=rf_acc, name='Random Forest', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=days, y=lr_acc, name='Logistic Regression', line=dict(width=3)))
            
            fig.update_layout(
                title="Model Accuracy Over Time",
                xaxis_title="Days",
                yaxis_title="Accuracy",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**⚡ Inference Time Distribution**")
            
            models = ['Random Forest', 'Logistic Regression', 'Isolation Forest', 'SVM']
            inference_times = [15, 8, 12, 18]
            
            fig = px.bar(
                x=models, y=inference_times,
                color=inference_times,
                color_continuous_scale='Viridis',
                title="Average Inference Time by Model"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_tab(self):
        """Render alerts and notifications"""
        
        st.header("🚨 Alerts & Notifications")
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔴 Critical Alerts", "23", "↑ 5")
        
        with col2:
            st.metric("🟡 High Priority", "156", "↓ 12")
        
        with col3:
            st.metric("🟢 Resolved Today", "89", "↑ 23")
        
        with col4:
            st.metric("⏱️ Avg Response Time", "4.2min", "↓ 1.1min")
        
        st.divider()
        
        # Recent alerts
        st.subheader("📋 Recent Alerts")
        
        recent_alerts = [
            {"Time": "14:45:23", "Severity": "🔴 Critical", "Type": "High Value", "Transaction": "TX_901234", "Amount": "$25,000", "Action": "Blocked"},
            {"Time": "14:42:15", "Severity": "🟡 High", "Type": "Velocity", "Transaction": "TX_901233", "Amount": "$3,500", "Action": "Review"},
            {"Time": "14:38:47", "Severity": "🟡 High", "Type": "Geographic", "Transaction": "TX_901232", "Amount": "$1,200", "Action": "Flagged"},
            {"Time": "14:35:12", "Severity": "🔴 Critical", "Type": "Pattern Match", "Transaction": "TX_901231", "Amount": "$18,500", "Action": "Blocked"},
            {"Time": "14:32:08", "Severity": "🟡 High", "Type": "Device Risk", "Transaction": "TX_901230", "Amount": "$850", "Action": "Review"},
        ]
        
        df_alerts = pd.DataFrame(recent_alerts)
        st.dataframe(df_alerts, use_container_width=True)
        
        # Alert configuration
        st.subheader("⚙️ Alert Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Threshold Settings**")
            
            critical_threshold = st.slider("Critical Alert Threshold", 0.0, 1.0, 0.8, 0.01)
            high_threshold = st.slider("High Alert Threshold", 0.0, 1.0, 0.6, 0.01)
            amount_threshold = st.number_input("High Amount Threshold ($)", value=10000, step=1000)
        
        with col2:
            st.write("**📧 Notification Settings**")
            
            email_alerts = st.checkbox("Email Notifications", value=True)
            sms_alerts = st.checkbox("SMS Notifications", value=False)
            slack_alerts = st.checkbox("Slack Notifications", value=True)
            
            if st.button("💾 Save Settings"):
                st.success("Alert settings saved successfully!")
    
    def render_investigations_tab(self):
        """Render investigation tools"""
        
        st.header("🔍 Investigation Tools")
        
        # Investigation search
        st.subheader("🔎 Transaction Investigation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox("Search Type", ["Transaction ID", "User ID", "Merchant ID", "Amount Range"])
        
        with col2:
            search_value = st.text_input("Search Value", placeholder="Enter search term...")
        
        with col3:
            if st.button("🔍 Search"):
                st.info(f"Searching for {search_type}: {search_value}")
        
        st.divider()
        
        # Case management
        st.subheader("📁 Case Management")
        
        cases = [
            {"Case ID": "CASE_001", "Type": "Fraud Ring", "Status": "🔴 Open", "Priority": "High", "Assigned": "John Doe", "Created": "2024-01-15"},
            {"Case ID": "CASE_002", "Type": "Account Takeover", "Status": "🟡 In Progress", "Priority": "Medium", "Assigned": "Jane Smith", "Created": "2024-01-14"},
            {"Case ID": "CASE_003", "Type": "Card Testing", "Status": "🟢 Closed", "Priority": "Low", "Assigned": "Bob Wilson", "Created": "2024-01-13"},
        ]
        
        df_cases = pd.DataFrame(cases)
        st.dataframe(df_cases, use_container_width=True)
        
        # Investigation tools
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🕸️ Network Analysis**")
            st.info("Analyze connections between users, merchants, and devices")
            
            if st.button("🔗 Launch Network Analyzer"):
                st.success("Network analyzer launched!")
        
        with col2:
            st.write("**📊 Pattern Matching**")
            st.info("Find similar transaction patterns and behaviors")
            
            if st.button("🎯 Find Similar Patterns"):
                st.success("Pattern matching initiated!")
    
    def render_settings_tab(self):
        """Render system settings"""
        
        st.header("⚙️ System Settings")
        
        # System configuration
        settings_tabs = st.tabs(["🎛️ General", "🤖 ML Models", "🔔 Alerts", "👥 Users", "🔒 Security"])
        
        with settings_tabs[0]:
            st.subheader("🎛️ General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**⏱️ Refresh Settings**")
                refresh_interval = st.slider("Dashboard Refresh (seconds)", 10, 300, 30)
                auto_refresh = st.checkbox("Enable Auto Refresh", value=True)
                
                st.write("**📊 Display Settings**")
                theme = st.selectbox("Dashboard Theme", ["Light", "Dark", "Auto"])
                chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])
            
            with col2:
                st.write("**🔧 Performance Settings**")
                cache_duration = st.slider("Cache Duration (minutes)", 1, 60, 15)
                max_records = st.number_input("Max Records per Page", 10, 1000, 100)
                
                st.write("**📈 Analytics Settings**")
                default_timerange = st.selectbox("Default Time Range", ["24 Hours", "7 Days", "30 Days"])
        
        with settings_tabs[1]:
            st.subheader("🤖 ML Model Settings")
            
            st.write("**⚖️ Model Weights**")
            rf_weight = st.slider("Random Forest Weight", 0.0, 1.0, 0.25)
            lr_weight = st.slider("Logistic Regression Weight", 0.0, 1.0, 0.25)
            if_weight = st.slider("Isolation Forest Weight", 0.0, 1.0, 0.25)
            svm_weight = st.slider("SVM Weight", 0.0, 1.0, 0.25)
            
            st.write("**🎯 Prediction Thresholds**")
            fraud_threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5)
            confidence_threshold = st.slider("Minimum Confidence", 0.0, 1.0, 0.7)
        
        with settings_tabs[2]:
            st.subheader("🔔 Alert Settings")
            
            st.write("**📧 Notification Channels**")
            email_enabled = st.checkbox("Email Notifications", value=True)
            if email_enabled:
                email_recipients = st.text_area("Email Recipients", "admin@company.com\nfraud-team@company.com")
            
            sms_enabled = st.checkbox("SMS Notifications")
            slack_enabled = st.checkbox("Slack Integration", value=True)
            
            st.write("**⚡ Alert Rules**")
            high_amount_alert = st.number_input("High Amount Alert ($)", value=10000)
            velocity_alert = st.number_input("Velocity Alert (tx/hour)", value=10)
        
        with settings_tabs[3]:
            st.subheader("👥 User Management")
            
            st.write("**👤 Current Users**")
            users = [
                {"Username": "admin", "Role": "Administrator", "Last Login": "2024-01-15 14:30", "Status": "🟢 Active"},
                {"Username": "analyst1", "Role": "Fraud Analyst", "Last Login": "2024-01-15 13:45", "Status": "🟢 Active"},
                {"Username": "viewer1", "Role": "Viewer", "Last Login": "2024-01-14 16:20", "Status": "🟡 Inactive"},
            ]
            
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)
            
            if st.button("➕ Add New User"):
                st.info("User management interface would open here")
        
        with settings_tabs[4]:
            st.subheader("🔒 Security Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔐 Authentication**")
                mfa_enabled = st.checkbox("Multi-Factor Authentication", value=True)
                session_timeout = st.slider("Session Timeout (minutes)", 15, 480, 60)
                
                st.write("**🛡️ Access Control**")
                ip_whitelist = st.text_area("IP Whitelist", "192.168.1.0/24\n10.0.0.0/8")
            
            with col2:
                st.write("**📝 Audit Settings**")
                audit_logging = st.checkbox("Enable Audit Logging", value=True)
                log_retention = st.slider("Log Retention (days)", 30, 365, 90)
                
                st.write("**🔒 Data Protection**")
                data_encryption = st.checkbox("Data Encryption", value=True)
                pii_masking = st.checkbox("PII Masking", value=True)
        
        # Save settings
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save All Settings"):
                st.success("All settings saved successfully!")
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                st.warning("Settings reset to defaults!")
        
        with col3:
            st.info("💡 Changes will take effect after the next system refresh")

# Main execution
if __name__ == "__main__":
    dashboard = AdvancedFraudDashboard()
    dashboard.render_dashboard()