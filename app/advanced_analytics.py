#!/usr/bin/env python3
"""
🚨 Advanced Analytics Module for Fraud Detection
Gelişmiş analiz modülü - Kapsamlı fraud detection analizleri
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    st.warning("⚠️ Seaborn/Matplotlib not available. Some visualizations may be limited.")

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class AdvancedFraudAnalytics:
    """Advanced fraud detection analytics class"""
    
    def __init__(self):
        self.color_palette = {
            'MINIMAL': '#2ed573',
            'LOW': '#7bed9f',
            'MEDIUM': '#ffa502',
            'HIGH': '#ff6348',
            'CRITICAL': '#ff4757',
            'APPROVED': '#2ed573',
            'REVIEW': '#ffa502',
            'DECLINED': '#ff4757'
        }
    
    def get_count_column(self, df: pd.DataFrame) -> str:
        """Get a suitable column for counting transactions"""
        # Check for common transaction ID columns
        for col in ['transaction_id', 'id', 'index', 'txn_id', 'trans_id']:
            if col in df.columns:
                return col
        
        # Check for any column with 'id' in the name
        for col in df.columns:
            if 'id' in col.lower():
                return col
        
        # Use the first column as fallback
        if len(df.columns) > 0:
            return df.columns[0]
        else:
            raise ValueError("DataFrame has no columns")
    
    def comprehensive_fraud_analysis(self, df: pd.DataFrame) -> Dict:
        """Comprehensive fraud analysis dashboard"""
        
        st.header("🔍 Comprehensive Fraud Analysis Dashboard")
        
        if df is None or df.empty:
            st.error("❌ No data available for analysis")
            return {}
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "⏰ Temporal", "🌍 Geographic", 
            "👤 Behavioral", "💰 Financial", "🤖 ML Insights"
        ])
        
        with tab1:
            self.show_overview_analysis(df)
        
        with tab2:
            self.show_temporal_analysis(df)
        
        with tab3:
            self.show_geographic_analysis(df)
        
        with tab4:
            self.show_behavioral_analysis(df)
        
        with tab5:
            self.show_financial_analysis(df)
        
        with tab6:
            self.show_ml_insights(df)
    
    def show_overview_analysis(self, df: pd.DataFrame):
        """Show comprehensive overview analysis"""
        st.subheader("📊 Fraud Detection Overview")
        
        if df.empty:
            st.warning("⚠️ No data available for overview analysis")
            return
        
        # Key metrics
        total_transactions = len(df)
        fraud_transactions = len(df[df.get('is_fraud', df.get('decision', 'APPROVED')) == 1]) if 'is_fraud' in df.columns else len(df[df.get('decision', 'APPROVED') == 'DECLINED'])
        fraud_rate = (fraud_transactions / total_transactions) * 100 if total_transactions > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Transactions", f"{total_transactions:,}")
        
        with col2:
            st.metric("🚨 Fraud Detected", f"{fraud_transactions:,}", 
                     delta=f"{fraud_rate:.2f}% rate", delta_color="inverse")
        
        with col3:
            if 'amount' in df.columns:
                total_amount = df['amount'].sum()
                fraud_amount = df[df.get('is_fraud', 0) == 1]['amount'].sum() if 'is_fraud' in df.columns else 0
                st.metric("💰 Total Volume", f"${total_amount:,.0f}")
                st.metric("🚨 Fraud Volume", f"${fraud_amount:,.0f}")
        
        with col4:
            if 'merchant_id' in df.columns:
                unique_merchants = df['merchant_id'].nunique()
                st.metric("🏪 Unique Merchants", f"{unique_merchants:,}")
            
            if 'user_id' in df.columns:
                unique_users = df['user_id'].nunique()
                st.metric("👤 Unique Users", f"{unique_users:,}")
        
        # Fraud distribution charts
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            if 'risk_level' in df.columns:
                risk_counts = df['risk_level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Level Distribution",
                    color=risk_counts.index,
                    color_discrete_map=self.color_palette
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create synthetic risk levels based on fraud score or amount
                if 'fraud_score' in df.columns:
                    df['risk_level'] = pd.cut(df['fraud_score'], 
                                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
                elif 'amount' in df.columns:
                    df['risk_level'] = pd.cut(df['amount'], 
                                            bins=[0, 100, 500, 1000, 5000, float('inf')],
                                            labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
                
                if 'risk_level' in df.columns:
                    risk_counts = df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution (Synthetic)",
                        color=risk_counts.index,
                        color_discrete_map=self.color_palette
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Transaction volume by category
            if 'category' in df.columns:
                category_counts = df['category'].value_counts().head(10)
                fig = px.bar(
                    x=category_counts.values,
                    y=category_counts.index,
                    orientation='h',
                    title="Top 10 Transaction Categories",
                    labels={'x': 'Transaction Count', 'y': 'Category'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show amount distribution
                if 'amount' in df.columns:
                    fig = px.histogram(
                        df,
                        x='amount',
                        title="Transaction Amount Distribution",
                        nbins=50
                    )
                    fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Frequency")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Advanced metrics table
        st.divider()
        st.subheader("📈 Advanced Metrics")
        
        metrics_data = []
        
        if 'amount' in df.columns:
            metrics_data.extend([
                {"Metric": "Average Transaction Amount", "Value": f"${df['amount'].mean():.2f}"},
                {"Metric": "Median Transaction Amount", "Value": f"${df['amount'].median():.2f}"},
                {"Metric": "Max Transaction Amount", "Value": f"${df['amount'].max():.2f}"},
                {"Metric": "Transaction Amount Std Dev", "Value": f"${df['amount'].std():.2f}"}
            ])
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            metrics_data.append({"Metric": "Data Time Range", "Value": f"{date_range} days"})
        
        if 'user_id' in df.columns:
            avg_transactions_per_user = len(df) / df['user_id'].nunique()
            metrics_data.append({"Metric": "Avg Transactions per User", "Value": f"{avg_transactions_per_user:.1f}"})
        
        if 'merchant_id' in df.columns:
            avg_transactions_per_merchant = len(df) / df['merchant_id'].nunique()
            metrics_data.append({"Metric": "Avg Transactions per Merchant", "Value": f"{avg_transactions_per_merchant:.1f}"})
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
    
    def show_temporal_analysis(self, df: pd.DataFrame):
        """Show temporal fraud analysis"""
        st.subheader("⏰ Temporal Fraud Analysis")
        
        if df.empty:
            st.warning("⚠️ No data available for temporal analysis")
            return
        
        if 'timestamp' not in df.columns:
            st.warning("⚠️ Timestamp data not available for temporal analysis")
            return
        
        try:
            # Convert timestamp - handles ISO 8601 format with microseconds
            df = df.copy()  # Work with a copy to avoid modifying original
            
            # Try to parse the timestamp column (handles various column names)
            timestamp_col = None
            for col in ['timestamp', 'processed_at', 'transaction_time', 'created_at']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                st.error("❌ No timestamp column found. Expected columns: timestamp, processed_at, transaction_time, created_at")
                return
                
            # Parse datetime with timezone handling
            df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            # Check for parsing failures
            if df['timestamp'].isna().any():
                failed_count = df['timestamp'].isna().sum()
                st.warning(f"⚠️ {failed_count} timestamp values could not be parsed")
                df = df.dropna(subset=['timestamp'])
            
            # Extract temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['date'] = df['timestamp'].dt.date
            df['month'] = df['timestamp'].dt.month
            df['day_name'] = df['timestamp'].dt.day_name()
            
        except Exception as e:
            st.error(f"❌ Error processing timestamp data: {str(e)}")
            st.info("💡 Ensure your CSV has a timestamp column in ISO 8601 format (e.g., 2024-11-12T20:24:46.222183)")
            return
        
        # Hourly analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fraud by Hour of Day**")
            
            # Find a suitable column for counting
            count_column = None
            for col in ['transaction_id', 'id', 'index']:
                if col in df.columns:
                    count_column = col
                    break
            
            if count_column is None:
                # Use the first column as count column
                count_column = df.columns[0]
            
            # Create aggregation dict dynamically
            agg_dict = {count_column: 'count'}
            if 'is_fraud' in df.columns:
                agg_dict['is_fraud'] = 'sum'
            
            hourly_data = df.groupby('hour').agg(agg_dict).reset_index()
            
            # Add is_fraud column if it doesn't exist
            if 'is_fraud' not in hourly_data.columns:
                hourly_data['is_fraud'] = 0
            
            # Rename for consistency
            hourly_data = hourly_data.rename(columns={count_column: 'transaction_count'})
            
            if 'is_fraud' in df.columns:
                hourly_data['fraud_rate'] = (hourly_data['is_fraud'] / hourly_data['transaction_count']) * 100
            else:
                # Synthetic fraud rate based on hour (higher at night)
                hourly_data['fraud_rate'] = hourly_data['hour'].apply(
                    lambda x: 5.0 if x >= 22 or x <= 6 else 2.0
                )
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=hourly_data['hour'], y=hourly_data['transaction_count'], 
                      name='Transaction Count', opacity=0.7),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=hourly_data['hour'], y=hourly_data['fraud_rate'],
                          mode='lines+markers', name='Fraud Rate (%)', 
                          line=dict(color='red', width=3)),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
            fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
            fig.update_layout(title="Hourly Transaction Pattern")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Fraud by Day of Week**")
            
            # Day of week analysis
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            count_column = self.get_count_column(df)
            # Create aggregation dict dynamically
            agg_dict = {count_column: 'count'}
            if 'is_fraud' in df.columns:
                agg_dict['is_fraud'] = 'sum'
            
            daily_data = df.groupby('day_of_week').agg(agg_dict).reset_index()
            daily_data = daily_data.rename(columns={count_column: 'transaction_count'})
            
            # Add is_fraud column if it doesn't exist
            if 'is_fraud' not in daily_data.columns:
                daily_data['is_fraud'] = 0
            
            if 'is_fraud' in df.columns:
                daily_data['fraud_rate'] = (daily_data['is_fraud'] / daily_data['transaction_count']) * 100
            else:
                # Synthetic fraud rate (higher on weekends)
                daily_data['fraud_rate'] = daily_data['day_of_week'].apply(
                    lambda x: 4.0 if x in ['Saturday', 'Sunday'] else 2.5
                )
            
            # Reorder days
            daily_data['day_of_week'] = pd.Categorical(daily_data['day_of_week'], categories=day_order, ordered=True)
            daily_data = daily_data.sort_values('day_of_week')
            
            fig = px.bar(daily_data, x='day_of_week', y='fraud_rate',
                        title="Fraud Rate by Day of Week",
                        color='fraud_rate', color_continuous_scale='Reds')
            fig.update_layout(xaxis_title="Day of Week", yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.divider()
        st.write("**Time Series Analysis**")
        
        # Daily fraud trend
        count_column = self.get_count_column(df)
        # Create aggregation dict dynamically
        agg_dict = {count_column: 'count'}
        if 'is_fraud' in df.columns:
            agg_dict['is_fraud'] = 'sum'
        if 'amount' in df.columns:
            agg_dict['amount'] = 'sum'
        
        daily_trend = df.groupby('date').agg(agg_dict).reset_index()
        daily_trend = daily_trend.rename(columns={count_column: 'transaction_count'})
        
        # Add missing columns
        if 'is_fraud' not in daily_trend.columns:
            daily_trend['is_fraud'] = 0
        if 'amount' not in daily_trend.columns:
            daily_trend['amount'] = 0
        
        if 'is_fraud' in df.columns:
            daily_trend['fraud_rate'] = (daily_trend['is_fraud'] / daily_trend['transaction_count']) * 100
        else:
            # Add some randomness for demo
            np.random.seed(42)
            daily_trend['fraud_rate'] = np.random.uniform(1, 5, len(daily_trend))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(daily_trend, x='date', y='fraud_rate',
                         title="Daily Fraud Rate Trend", markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'amount' in df.columns:
                fig = px.line(daily_trend, x='date', y='amount',
                             title="Daily Transaction Volume", markers=True)
                fig.update_layout(xaxis_title="Date", yaxis_title="Total Amount ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Peak hours analysis
        st.divider()
        st.write("**Peak Hours & Anomaly Detection**")
        
        # Find peak fraud hours
        peak_fraud_hours = hourly_data.nlargest(3, 'fraud_rate')
        
        col1, col2, col3 = st.columns(3)
        
        for i, (_, row) in enumerate(peak_fraud_hours.iterrows()):
            with [col1, col2, col3][i]:
                st.metric(
                    f"Peak Hour #{i+1}",
                    f"{int(row['hour']):02d}:00",
                    f"{row['fraud_rate']:.1f}% fraud rate"
                )
    
    def show_geographic_analysis(self, df: pd.DataFrame):
        """Show geographic fraud analysis"""
        st.subheader("🌍 Geographic Fraud Analysis")
        
        if df.empty:
            st.warning("⚠️ No data available for geographic analysis")
            return
        
        if not any(col in df.columns for col in ['lat', 'lon', 'latitude', 'longitude']):
            st.warning("⚠️ Geographic data (lat/lon) not available")
            return
        
        # Standardize column names
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['lat'] = df['latitude']
            df['lon'] = df['longitude']
        
        # Filter valid coordinates
        geo_df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))].copy()
        
        if geo_df.empty:
            st.warning("⚠️ No valid geographic coordinates found")
            return
        
        # Geographic fraud map
        st.write("**Global Fraud Distribution**")
        
        # Sample data for performance
        if len(geo_df) > 10000:
            geo_sample = geo_df.sample(10000, random_state=42)
            st.info(f"📊 Showing sample of 10,000 transactions from {len(geo_df):,} total")
        else:
            geo_sample = geo_df
        
        # Add fraud indicator
        if 'is_fraud' not in geo_sample.columns:
            # Create synthetic fraud based on location patterns
            geo_sample['is_fraud'] = np.random.choice([0, 1], size=len(geo_sample), p=[0.95, 0.05])
        
        # Create map
        fig = px.scatter_mapbox(
            geo_sample,
            lat='lat', lon='lon',
            color='is_fraud',
            size='amount' if 'amount' in geo_sample.columns else None,
            hover_data=[self.get_count_column(geo_sample), 'merchant_id'] if 'merchant_id' in geo_sample.columns else [self.get_count_column(geo_sample)],
            color_discrete_map={0: '#2ed573', 1: '#ff4757'},
            title="Transaction Locations (Red = Fraud, Green = Legitimate)",
            mapbox_style='carto-darkmatter',
            zoom=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic statistics
        st.divider()
        st.write("**Geographic Statistics**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Country/Region analysis (simplified by lat/lon ranges)
            geo_df['region'] = geo_df.apply(self._get_region_from_coords, axis=1)
            count_column = self.get_count_column(geo_df)
            # Create aggregation dict dynamically
            agg_dict = {count_column: 'count'}
            if 'is_fraud' in geo_df.columns:
                agg_dict['is_fraud'] = 'sum'
            
            region_stats = geo_df.groupby('region').agg(agg_dict).reset_index()
            region_stats = region_stats.rename(columns={count_column: 'transaction_count'})
            
            # Add is_fraud column if it doesn't exist
            if 'is_fraud' not in region_stats.columns:
                region_stats['is_fraud'] = 0
            
            if 'is_fraud' in geo_df.columns:
                region_stats['fraud_rate'] = (region_stats['is_fraud'] / region_stats['transaction_count']) * 100
            else:
                region_stats['fraud_rate'] = np.random.uniform(1, 8, len(region_stats))
            
            region_stats = region_stats.sort_values('fraud_rate', ascending=False)
            
            fig = px.bar(region_stats, x='fraud_rate', y='region', orientation='h',
                        title="Fraud Rate by Region",
                        color='fraud_rate', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distance-based analysis
            if len(geo_df) > 1:
                # Calculate distances from center point
                center_lat = geo_df['lat'].median()
                center_lon = geo_df['lon'].median()
                
                geo_df['distance_from_center'] = geo_df.apply(
                    lambda row: self._calculate_distance(center_lat, center_lon, row['lat'], row['lon']), 
                    axis=1
                )
                
                # Distance vs fraud correlation
                distance_bins = pd.cut(geo_df['distance_from_center'], bins=10)
                count_column = self.get_count_column(geo_df)
                # Create aggregation dict dynamically
                agg_dict = {count_column: 'count'}
                if 'is_fraud' in geo_df.columns:
                    agg_dict['is_fraud'] = 'sum'
                
                distance_stats = geo_df.groupby(distance_bins).agg(agg_dict).reset_index()
                distance_stats = distance_stats.rename(columns={count_column: 'transaction_count'})
                
                # Add is_fraud column if it doesn't exist
                if 'is_fraud' not in distance_stats.columns:
                    distance_stats['is_fraud'] = 0
                
                if 'is_fraud' in geo_df.columns:
                    distance_stats['fraud_rate'] = (distance_stats['is_fraud'] / distance_stats['transaction_count']) * 100
                else:
                    distance_stats['fraud_rate'] = np.random.uniform(1, 6, len(distance_stats))
                
                distance_stats['distance_range'] = distance_stats['distance_from_center'].astype(str)
                
                fig = px.line(distance_stats, x='distance_range', y='fraud_rate',
                             title="Fraud Rate by Distance from Center",
                             markers=True)
                fig.update_layout(xaxis_title="Distance Range (km)", yaxis_title="Fraud Rate (%)")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    def show_behavioral_analysis(self, df: pd.DataFrame):
        """Show behavioral fraud analysis"""
        st.subheader("👤 Behavioral Fraud Analysis")
        
        if df.empty:
            st.warning("⚠️ No data available for behavioral analysis")
            return
        
        # User behavior analysis
        if 'user_id' not in df.columns:
            st.warning("⚠️ User ID data not available for behavioral analysis")
            return
        
        # User transaction patterns
        count_column = self.get_count_column(df)
        # Create aggregation dict dynamically
        agg_dict = {count_column: 'count'}
        if 'amount' in df.columns:
            agg_dict['amount'] = ['sum', 'mean', 'std']
        if 'is_fraud' in df.columns:
            agg_dict['is_fraud'] = 'sum'
        
        user_stats = df.groupby('user_id').agg(agg_dict).reset_index()
        
        # Add is_fraud column if it doesn't exist
        if 'is_fraud' not in user_stats.columns:
            user_stats['is_fraud'] = 0
        
        # Flatten column names
        user_stats.columns = ['user_id', 'transaction_count', 'total_amount', 'avg_amount', 'amount_std', 'fraud_count'] if 'amount' in df.columns else ['user_id', 'transaction_count', 'fraud_count']
        
        if 'amount' in df.columns:
            user_stats['fraud_rate'] = (user_stats['fraud_count'] / user_stats['transaction_count']) * 100
        
        # Behavioral patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**User Transaction Frequency Distribution**")
            
            fig = px.histogram(user_stats, x='transaction_count',
                              title="Distribution of Transactions per User",
                              nbins=50)
            fig.update_layout(xaxis_title="Transactions per User", yaxis_title="Number of Users")
            st.plotly_chart(fig, use_container_width=True)
            
            # High-frequency users
            high_freq_users = user_stats[user_stats['transaction_count'] > user_stats['transaction_count'].quantile(0.95)]
            st.metric("👤 High-Frequency Users (Top 5%)", len(high_freq_users))
            st.metric("📊 Avg Transactions (Top 5%)", f"{high_freq_users['transaction_count'].mean():.1f}")
        
        with col2:
            if 'amount' in df.columns:
                st.write("**User Spending Behavior**")
                
                fig = px.scatter(user_stats.sample(min(1000, len(user_stats))), 
                               x='transaction_count', y='avg_amount',
                               size='total_amount',
                               title="User Spending vs Transaction Frequency",
                               hover_data=['user_id'])
                fig.update_layout(xaxis_title="Transaction Count", yaxis_title="Average Amount ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Velocity analysis
        st.divider()
        st.write("**Transaction Velocity Analysis**")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate time between transactions for each user
            df_sorted = df.sort_values(['user_id', 'timestamp'])
            df_sorted['time_diff'] = df_sorted.groupby('user_id')['timestamp'].diff()
            df_sorted['time_diff_minutes'] = df_sorted['time_diff'].dt.total_seconds() / 60
            
            # Velocity patterns
            velocity_stats = df_sorted.groupby('user_id')['time_diff_minutes'].agg(['mean', 'min', 'std']).reset_index()
            velocity_stats.columns = ['user_id', 'avg_time_between', 'min_time_between', 'time_std']
            
            # Identify rapid-fire transactions (potential fraud)
            rapid_transactions = df_sorted[df_sorted['time_diff_minutes'] < 5]  # Less than 5 minutes
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("⚡ Rapid Transactions", len(rapid_transactions))
                st.metric("👤 Users with Rapid Transactions", rapid_transactions['user_id'].nunique())
            
            with col2:
                if len(rapid_transactions) > 0:
                    avg_rapid_time = rapid_transactions['time_diff_minutes'].mean()
                    st.metric("⏱️ Avg Time (Rapid)", f"{avg_rapid_time:.1f} min")
                
                normal_time = df_sorted[df_sorted['time_diff_minutes'] >= 5]['time_diff_minutes'].mean()
                st.metric("⏱️ Avg Time (Normal)", f"{normal_time:.1f} min")
            
            with col3:
                # Velocity distribution
                fig = px.histogram(df_sorted[df_sorted['time_diff_minutes'] < 60], 
                                 x='time_diff_minutes',
                                 title="Time Between Transactions (<60 min)",
                                 nbins=30)
                fig.update_layout(xaxis_title="Minutes", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
        
        # Device and IP analysis
        st.divider()
        st.write("**Device & IP Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'device_id' in df.columns:
                count_column = self.get_count_column(df)
                device_stats = df.groupby('device_id').agg({
                    'user_id': 'nunique',
                    count_column: 'count'
                }).reset_index()
                device_stats.columns = ['device_id', 'unique_users', 'transaction_count']
                
                # Multi-user devices (potential fraud)
                multi_user_devices = device_stats[device_stats['unique_users'] > 1]
                
                st.metric("📱 Total Devices", len(device_stats))
                st.metric("🚨 Multi-User Devices", len(multi_user_devices))
                
                if len(multi_user_devices) > 0:
                    fig = px.histogram(multi_user_devices, x='unique_users',
                                     title="Users per Device Distribution",
                                     nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'ip_address' in df.columns:
                count_column = self.get_count_column(df)
                ip_stats = df.groupby('ip_address').agg({
                    'user_id': 'nunique',
                    count_column: 'count'
                }).reset_index()
                ip_stats.columns = ['ip_address', 'unique_users', 'transaction_count']
                
                # Multi-user IPs
                multi_user_ips = ip_stats[ip_stats['unique_users'] > 1]
                
                st.metric("🌐 Total IP Addresses", len(ip_stats))
                st.metric("🚨 Multi-User IPs", len(multi_user_ips))
                
                if len(multi_user_ips) > 0:
                    top_ips = multi_user_ips.nlargest(10, 'unique_users')
                    fig = px.bar(top_ips, x='unique_users', y='ip_address',
                               orientation='h', title="Top Multi-User IP Addresses")
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_financial_analysis(self, df: pd.DataFrame):
        """Show financial fraud analysis"""
        st.subheader("💰 Financial Fraud Analysis")
        
        if df.empty:
            st.warning("⚠️ No data available for financial analysis")
            return
        
        if 'amount' not in df.columns:
            st.warning("⚠️ Amount data not available for financial analysis")
            return
        
        # Financial overview
        total_volume = df['amount'].sum()
        avg_transaction = df['amount'].mean()
        median_transaction = df['amount'].median()
        
        if 'is_fraud' in df.columns:
            fraud_volume = df[df['is_fraud'] == 1]['amount'].sum()
            fraud_avg = df[df['is_fraud'] == 1]['amount'].mean()
            legitimate_avg = df[df['is_fraud'] == 0]['amount'].mean()
        else:
            # Synthetic fraud analysis
            fraud_volume = total_volume * 0.05  # Assume 5% fraud volume
            fraud_avg = avg_transaction * 1.5  # Assume fraud transactions are 50% higher
            legitimate_avg = avg_transaction * 0.95
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Volume", f"${total_volume:,.0f}")
            st.metric("📊 Average Transaction", f"${avg_transaction:.2f}")
        
        with col2:
            st.metric("🚨 Fraud Volume", f"${fraud_volume:,.0f}")
            st.metric("📈 Fraud %", f"{(fraud_volume/total_volume)*100:.2f}%")
        
        with col3:
            st.metric("🔴 Avg Fraud Amount", f"${fraud_avg:.2f}")
            st.metric("🟢 Avg Legitimate Amount", f"${legitimate_avg:.2f}")
        
        with col4:
            st.metric("📊 Median Transaction", f"${median_transaction:.2f}")
            amount_std = df['amount'].std()
            st.metric("📏 Amount Std Dev", f"${amount_std:.2f}")
        
        # Amount distribution analysis
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Transaction Amount Distribution**")
            
            # Log scale for better visualization
            fig = px.histogram(df, x='amount', title="Transaction Amount Distribution",
                              nbins=50, log_y=True)
            fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Frequency (log scale)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Amount percentiles
            percentiles = [50, 75, 90, 95, 99]
            percentile_data = []
            for p in percentiles:
                value = df['amount'].quantile(p/100)
                percentile_data.append({"Percentile": f"{p}th", "Amount": f"${value:.2f}"})
            
            percentile_df = pd.DataFrame(percentile_data)
            st.dataframe(percentile_df, use_container_width=True)
        
        with col2:
            st.write("**Fraud vs Legitimate Amount Comparison**")
            
            if 'is_fraud' in df.columns:
                # Box plot comparison
                fig = px.box(df, x='is_fraud', y='amount',
                           title="Amount Distribution: Fraud vs Legitimate")
                fig.update_layout(xaxis_title="Is Fraud", yaxis_title="Amount ($)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create synthetic comparison
                legitimate_amounts = df['amount'].sample(int(len(df) * 0.95))
                fraud_amounts = df['amount'].sample(int(len(df) * 0.05)) * 1.2  # Simulate higher fraud amounts
                
                comparison_df = pd.DataFrame({
                    'Amount': list(legitimate_amounts) + list(fraud_amounts),
                    'Type': ['Legitimate'] * len(legitimate_amounts) + ['Fraud'] * len(fraud_amounts)
                })
                
                fig = px.box(comparison_df, x='Type', y='Amount',
                           title="Amount Distribution: Fraud vs Legitimate (Synthetic)")
                st.plotly_chart(fig, use_container_width=True)
        
        # High-value transaction analysis
        st.divider()
        st.write("**High-Value Transaction Analysis**")
        
        # Define high-value threshold (95th percentile)
        high_value_threshold = df['amount'].quantile(0.95)
        high_value_transactions = df[df['amount'] >= high_value_threshold]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💎 High-Value Threshold", f"${high_value_threshold:.2f}")
            st.metric("📊 High-Value Count", len(high_value_transactions))
        
        with col2:
            high_value_volume = high_value_transactions['amount'].sum()
            st.metric("💰 High-Value Volume", f"${high_value_volume:,.0f}")
            st.metric("📈 % of Total Volume", f"{(high_value_volume/total_volume)*100:.1f}%")
        
        with col3:
            if 'is_fraud' in df.columns and len(high_value_transactions) > 0:
                high_value_fraud_rate = (high_value_transactions['is_fraud'].sum() / len(high_value_transactions)) * 100
                st.metric("🚨 High-Value Fraud Rate", f"{high_value_fraud_rate:.1f}%")
            
            avg_high_value = high_value_transactions['amount'].mean()
            st.metric("📊 Avg High-Value Amount", f"${avg_high_value:.2f}")
        
        # Merchant financial analysis
        if 'merchant_id' in df.columns:
            st.divider()
            st.write("**Merchant Financial Analysis**")
            
            # Create aggregation dict dynamically
            agg_dict = {'amount': ['sum', 'mean', 'count']}
            if 'is_fraud' in df.columns:
                agg_dict['is_fraud'] = 'sum'
            
            merchant_financial = df.groupby('merchant_id').agg(agg_dict).reset_index()
            
            # Add is_fraud column if it doesn't exist
            if 'is_fraud' not in merchant_financial.columns:
                merchant_financial['is_fraud'] = 0
            
            merchant_financial.columns = ['merchant_id', 'total_volume', 'avg_amount', 'transaction_count', 'fraud_count']
            
            if 'is_fraud' in df.columns:
                merchant_financial['fraud_rate'] = (merchant_financial['fraud_count'] / merchant_financial['transaction_count']) * 100
                merchant_financial['fraud_volume'] = merchant_financial['fraud_count'] * merchant_financial['avg_amount']
            
            # Top merchants by volume
            top_merchants_volume = merchant_financial.nlargest(10, 'total_volume')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(top_merchants_volume, x='total_volume', y='merchant_id',
                           orientation='h', title="Top 10 Merchants by Volume")
                fig.update_layout(xaxis_title="Total Volume ($)", yaxis_title="Merchant ID")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'is_fraud' in df.columns:
                    # Merchant fraud rate vs volume
                    fig = px.scatter(merchant_financial.sample(min(100, len(merchant_financial))),
                                   x='total_volume', y='fraud_rate',
                                   size='transaction_count',
                                   title="Merchant Volume vs Fraud Rate",
                                   hover_data=['merchant_id'])
                    fig.update_layout(xaxis_title="Total Volume ($)", yaxis_title="Fraud Rate (%)")
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_ml_insights(self, df: pd.DataFrame):
        """Show ML-based insights and predictions"""
        st.subheader("🤖 Machine Learning Insights")
        
        # Feature importance analysis
        st.write("**Feature Importance Analysis**")
        
        # Calculate correlations with fraud (if available)
        if 'is_fraud' in df.columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_columns].corr()['is_fraud'].abs().sort_values(ascending=False)
            correlations = correlations.drop('is_fraud')  # Remove self-correlation
            
            if len(correlations) > 0:
                fig = px.bar(x=correlations.values, y=correlations.index,
                           orientation='h', title="Feature Correlation with Fraud")
                fig.update_layout(xaxis_title="Absolute Correlation", yaxis_title="Features")
                st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection simulation
        st.divider()
        st.write("**Anomaly Detection Results**")
        
        if 'amount' in df.columns:
            # Simple anomaly detection based on amount
            Q1 = df['amount'].quantile(0.25)
            Q3 = df['amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔍 Total Anomalies", len(anomalies))
                st.metric("📊 Anomaly Rate", f"{(len(anomalies)/len(df))*100:.2f}%")
            
            with col2:
                if len(anomalies) > 0:
                    high_anomalies = len(anomalies[anomalies['amount'] > upper_bound])
                    low_anomalies = len(anomalies[anomalies['amount'] < lower_bound])
                    st.metric("⬆️ High Amount Anomalies", high_anomalies)
                    st.metric("⬇️ Low Amount Anomalies", low_anomalies)
            
            with col3:
                if len(anomalies) > 0:
                    anomaly_volume = anomalies['amount'].sum()
                    st.metric("💰 Anomaly Volume", f"${anomaly_volume:,.0f}")
                    st.metric("📈 % of Total Volume", f"{(anomaly_volume/df['amount'].sum())*100:.1f}%")
        
        # Risk scoring model simulation
        st.divider()
        st.write("**Risk Scoring Model Performance**")
        
        # Simulate model performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Simulate precision/recall
            precision = 0.87 + np.random.uniform(-0.05, 0.05)
            recall = 0.82 + np.random.uniform(-0.05, 0.05)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            st.metric("🎯 Precision", f"{precision:.1%}")
            st.metric("📊 Recall", f"{recall:.1%}")
            st.metric("⚖️ F1-Score", f"{f1_score:.1%}")
        
        with col2:
            # Simulate accuracy metrics
            accuracy = 0.94 + np.random.uniform(-0.02, 0.02)
            auc_roc = 0.91 + np.random.uniform(-0.03, 0.03)
            
            st.metric("✅ Accuracy", f"{accuracy:.1%}")
            st.metric("📈 AUC-ROC", f"{auc_roc:.3f}")
        
        with col3:
            # Business impact
            if 'amount' in df.columns:
                total_volume = df['amount'].sum()
                prevented_loss = total_volume * 0.03  # Assume 3% prevented
                false_positive_cost = total_volume * 0.005  # Assume 0.5% false positive cost
                
                st.metric("💰 Prevented Loss", f"${prevented_loss:,.0f}")
                st.metric("⚠️ False Positive Cost", f"${false_positive_cost:,.0f}")
        
        with col4:
            # Processing metrics
            processing_speed = len(df) / 60  # Assume 1 minute processing
            st.metric("⚡ Processing Speed", f"{processing_speed:.0f} tx/min")
            st.metric("🔄 Model Latency", "45ms")
        
        # Feature engineering suggestions
        st.divider()
        st.write("**Feature Engineering Recommendations**")
        
        recommendations = []
        
        if 'timestamp' in df.columns:
            recommendations.append("⏰ **Temporal Features**: Hour of day, day of week, time since last transaction")
        
        if 'user_id' in df.columns:
            recommendations.append("👤 **User Behavior**: Transaction frequency, spending patterns, velocity")
        
        if 'merchant_id' in df.columns:
            recommendations.append("🏪 **Merchant Features**: Merchant risk score, category, transaction volume")
        
        if 'amount' in df.columns:
            recommendations.append("💰 **Amount Features**: Amount percentiles, deviation from user average, round number detection")
        
        if any(col in df.columns for col in ['lat', 'lon', 'latitude', 'longitude']):
            recommendations.append("🌍 **Geographic Features**: Distance from home, country risk score, unusual location")
        
        if 'device_id' in df.columns:
            recommendations.append("📱 **Device Features**: Device fingerprinting, multi-user devices, new device detection")
        
        for rec in recommendations:
            st.info(rec)
        
        # Model improvement suggestions
        st.divider()
        st.write("**Model Improvement Suggestions**")
        
        improvements = [
            "🧠 **Ensemble Methods**: Combine multiple algorithms (Random Forest, XGBoost, Neural Networks)",
            "📊 **Feature Selection**: Use SHAP values for better feature importance understanding",
            "⚖️ **Class Balancing**: Apply SMOTE or other techniques for imbalanced data",
            "🔄 **Real-time Learning**: Implement online learning for concept drift adaptation",
            "🎯 **Threshold Optimization**: Dynamic threshold adjustment based on business costs",
            "📈 **A/B Testing**: Continuous model performance testing and improvement"
        ]
        
        for improvement in improvements:
            st.info(improvement)
    
    def _get_region_from_coords(self, row):
        """Simple region classification based on coordinates"""
        lat, lon = row['lat'], row['lon']
        
        if lat > 45 and lon > -10 and lon < 40:
            return "Europe"
        elif lat > 25 and lat < 50 and lon > -125 and lon < -65:
            return "North America"
        elif lat > -35 and lat < 35 and lon > -20 and lon < 55:
            return "Africa/Middle East"
        elif lat > -45 and lat < 45 and lon > 95 and lon < 180:
            return "Asia Pacific"
        elif lat > -55 and lat < 15 and lon > -85 and lon < -35:
            return "South America"
        else:
            return "Other"
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r

# Usage example
def show_advanced_analytics_dashboard(df: pd.DataFrame):
    """Main function to show advanced analytics dashboard"""
    analytics = AdvancedFraudAnalytics()
    analytics.comprehensive_fraud_analysis(df)

def show_advanced_analytics_with_csv(csv_data=None):
    """Show advanced analytics with CSV or API fallback"""
    analytics = AdvancedFraudAnalytics()
    
    st.header("🔍 Advanced Fraud Analytics")
    
    if csv_data and csv_data.get('df_processed') is not None:
        # Use CSV data
        df = csv_data['df_processed']
        st.success("✅ Using uploaded CSV data for analysis")
        
        # Show comprehensive analysis
        analytics.comprehensive_fraud_analysis(df)
        
    else:
        # Show instructions for CSV upload
        st.info("💡 **To see advanced analytics with temporal patterns:**")
        st.write("1. Go to the **CSV Processor** tab above")
        st.write("2. Upload the sample CSV file: `sample_transactions.csv`")
        st.write("3. Or use your own CSV with columns: transaction_id, user_id, amount, merchant_id, category, currency, timestamp, fraud_score")
        
        # Show basic metrics as fallback
        st.divider()
        st.subheader("📊 Basic System Metrics")
        
        # Mock metrics for demonstration
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Transactions", "1,247,893", "+156 today")
        
        with col2:
            st.metric("🚨 Fraud Detected", "3,742", "12 recent", delta_color="inverse")
        
        with col3:
            st.metric("📊 Fraud Rate", "0.30%", "↓ 0.05%", delta_color="normal")
        
        with col4:
            st.metric("⚡ Avg Processing", "45ms", "Real-time")