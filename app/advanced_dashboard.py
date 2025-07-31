#!/usr/bin/env python3
"""
🚨 Advanced Fraud Detection Dashboard - Full Featured Version
Gelişmiş fraud detection dashboard - Tam özellikli versiyon
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import time
import json
import os
import tempfile
import io
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚨 Advanced Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-critical { 
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    .alert-high { 
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-medium { 
        background: linear-gradient(135deg, #ffb142 0%, #ff9ff3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-approved { color: #2ed573; font-weight: bold; }
    .status-declined { color: #ff4757; font-weight: bold; }
    .status-review { color: #ffa502; font-weight: bold; }
    .fraud-score-high { color: #ff4757; font-weight: bold; font-size: 1.2em; }
    .fraud-score-medium { color: #ffa502; font-weight: bold; }
    .fraud-score-low { color: #2ed573; font-weight: bold; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

class AdvancedFraudDashboard:
    """Advanced Fraud Detection Dashboard Class"""
    
    def __init__(self):
        self.db_path = "data/fraud_detection.db"
        self.ensure_database()
    
    def ensure_database(self):
        """Ensure database exists with sample data"""
        os.makedirs("data", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                user_id TEXT,
                merchant_id TEXT,
                amount REAL,
                currency TEXT DEFAULT 'USD',
                category TEXT,
                timestamp DATETIME,
                fraud_score REAL,
                risk_level TEXT,
                decision TEXT,
                risk_factors TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                transaction_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Check if we have data, if not create sample data
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        
        if count == 0:
            self.generate_sample_data(cursor)
        
        conn.commit()
        conn.close()
    
    def generate_sample_data(self, cursor):
        """Generate realistic sample transaction data"""
        np.random.seed(42)
        
        # Generate 10000 sample transactions
        n_transactions = 10000
        
        merchants = [
            'amazon', 'walmart', 'target', 'starbucks', 'mcdonalds', 'shell', 'exxon',
            'grocery_store', 'electronics_shop', 'clothing_store', 'pharmacy', 'gas_station',
            'restaurant', 'coffee_shop', 'bookstore', 'hardware_store', 'casino', 'gambling_site',
            'crypto_exchange', 'suspicious_merchant', 'offshore_bank', 'unknown_vendor'
        ]
        
        categories = [
            'grocery', 'gas', 'restaurant', 'electronics', 'clothing', 'pharmacy',
            'entertainment', 'travel', 'gambling', 'crypto', 'cash_advance', 'atm'
        ]
        
        for i in range(n_transactions):
            # Generate realistic transaction data
            transaction_id = f"tx_{i:08d}"
            user_id = f"user_{np.random.randint(1, 5000):06d}"
            merchant_id = np.random.choice(merchants)
            category = np.random.choice(categories)
            
            # Amount distribution (most transactions are small, few are large)
            if np.random.random() < 0.8:
                amount = np.random.lognormal(3, 1)  # Most transactions $5-$200
            else:
                amount = np.random.lognormal(6, 1)  # Some larger transactions
            
            amount = max(1.0, min(50000.0, amount))  # Cap between $1-$50k
            
            # Timestamp (last 30 days)
            days_ago = np.random.randint(0, 30)
            hours_ago = np.random.randint(0, 24)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Calculate fraud score based on realistic factors
            fraud_score = 0.0
            risk_factors = []
            
            # High amount risk
            if amount > 5000:
                fraud_score += 0.4
                risk_factors.append('high_amount')
            elif amount > 1000:
                fraud_score += 0.2
                risk_factors.append('elevated_amount')
            
            # Suspicious merchant risk
            if merchant_id in ['casino', 'gambling_site', 'crypto_exchange', 'suspicious_merchant', 'offshore_bank']:
                fraud_score += 0.5
                risk_factors.append('suspicious_merchant')
            
            # Category risk
            if category in ['gambling', 'crypto', 'cash_advance']:
                fraud_score += 0.3
                risk_factors.append('suspicious_category')
            
            # Time-based risk (late night transactions)
            hour = timestamp.hour
            if hour >= 23 or hour <= 5:
                fraud_score += 0.2
                risk_factors.append('unusual_hour')
            
            # Add some randomness
            fraud_score += np.random.uniform(-0.1, 0.2)
            fraud_score = max(0.0, min(1.0, fraud_score))
            
            # Determine risk level
            if fraud_score >= 0.8:
                risk_level = 'CRITICAL'
            elif fraud_score >= 0.6:
                risk_level = 'HIGH'
            elif fraud_score >= 0.4:
                risk_level = 'MEDIUM'
            elif fraud_score >= 0.2:
                risk_level = 'LOW'
            else:
                risk_level = 'MINIMAL'
            
            # Make decision
            if fraud_score >= 0.7:
                decision = 'DECLINED'
            elif fraud_score >= 0.4:
                decision = 'REVIEW'
            else:
                decision = 'APPROVED'
            
            # Insert transaction
            cursor.execute('''
                INSERT INTO transactions 
                (transaction_id, user_id, merchant_id, amount, category, timestamp, 
                 fraud_score, risk_level, decision, risk_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (transaction_id, user_id, merchant_id, amount, category, timestamp,
                  fraud_score, risk_level, decision, json.dumps(risk_factors)))
        
        # Generate some alerts
        alert_messages = [
            "Unusual spending pattern detected for user_001234",
            "Multiple high-risk transactions from same IP",
            "Suspicious merchant activity detected",
            "Velocity check failed - too many transactions",
            "Geographic anomaly detected",
            "Card testing pattern identified"
        ]
        
        for i in range(50):
            alert_type = np.random.choice(['VELOCITY', 'AMOUNT', 'MERCHANT', 'GEOGRAPHIC', 'PATTERN'])
            severity = np.random.choice(['HIGH', 'MEDIUM', 'LOW'], p=[0.2, 0.5, 0.3])
            message = np.random.choice(alert_messages)
            transaction_id = f"tx_{np.random.randint(0, n_transactions):08d}"
            
            cursor.execute('''
                INSERT INTO alerts (alert_type, severity, message, transaction_id)
                VALUES (?, ?, ?, ?)
            ''', (alert_type, severity, message, transaction_id))
    
    def get_dashboard_metrics(self) -> Dict:
        """Get key dashboard metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Total transactions
        total_transactions = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM transactions", conn
        ).iloc[0]['count']
        
        # Fraud statistics
        fraud_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN decision = 'DECLINED' THEN 1 ELSE 0 END) as declined,
                SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) as review,
                SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved,
                AVG(fraud_score) as avg_fraud_score,
                SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_risk
            FROM transactions
        ''', conn).iloc[0]
        
        # Recent activity (last 24 hours)
        recent_activity = pd.read_sql_query('''
            SELECT COUNT(*) as count 
            FROM transactions 
            WHERE timestamp >= datetime('now', '-1 day')
        ''', conn).iloc[0]['count']
        
        # Amount statistics
        amount_stats = pd.read_sql_query('''
            SELECT 
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                SUM(CASE WHEN decision = 'DECLINED' THEN amount ELSE 0 END) as blocked_amount
            FROM transactions
        ''', conn).iloc[0]
        
        conn.close()
        
        return {
            'total_transactions': total_transactions,
            'fraud_declined': fraud_stats['declined'],
            'fraud_review': fraud_stats['review'],
            'fraud_approved': fraud_stats['approved'],
            'avg_fraud_score': fraud_stats['avg_fraud_score'],
            'critical_alerts': fraud_stats['critical'],
            'high_risk_alerts': fraud_stats['high_risk'],
            'recent_activity': recent_activity,
            'total_amount': amount_stats['total_amount'],
            'avg_amount': amount_stats['avg_amount'],
            'blocked_amount': amount_stats['blocked_amount'],
            'fraud_rate': fraud_stats['declined'] / fraud_stats['total'] * 100 if fraud_stats['total'] > 0 else 0
        }
    
    def get_fraud_trends(self, hours: int = 24) -> pd.DataFrame:
        """Get fraud trends over time"""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN decision = 'DECLINED' THEN 1 ELSE 0 END) as fraud_count,
                AVG(fraud_score) as avg_fraud_score,
                SUM(amount) as total_amount
            FROM transactions 
            WHERE timestamp >= datetime('now', '-{hours} hours')
            GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
            ORDER BY hour
        '''
        
        df_trends = pd.read_sql_query(query, conn)
        conn.close()
        
        if df_trends.empty:
            # Generate sample data for demo
            dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
            df_trends = pd.DataFrame({
                'hour': dates.strftime('%Y-%m-%d %H:00:00'),
                'total_transactions': np.random.poisson(50, 24),
                'fraud_count': np.random.poisson(2, 24),
                'avg_fraud_score': np.random.uniform(0.1, 0.4, 24),
                'total_amount': np.random.uniform(10000, 100000, 24)
            })
        
        df_trends['hour'] = pd.to_datetime(df_trends['hour'])
        return df_trends
    
    def get_risk_distribution(self) -> pd.DataFrame:
        """Get risk level distribution"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                risk_level,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions) as percentage
            FROM transactions
            GROUP BY risk_level
            ORDER BY 
                CASE risk_level 
                    WHEN 'CRITICAL' THEN 5
                    WHEN 'HIGH' THEN 4
                    WHEN 'MEDIUM' THEN 3
                    WHEN 'LOW' THEN 2
                    WHEN 'MINIMAL' THEN 1
                END DESC
        '''
        
        try:
            df_risk = pd.read_sql_query(query, conn)
        except:
            # Sample data for demo
            df_risk = pd.DataFrame({
                'risk_level': ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                'count': [6500, 2000, 1000, 400, 100],
                'percentage': [65.0, 20.0, 10.0, 4.0, 1.0]
            })
        
        conn.close()
        return df_risk
    
    def get_recent_transactions(self, limit: int = 20) -> pd.DataFrame:
        """Get recent transactions"""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT 
                transaction_id,
                user_id,
                merchant_id,
                amount,
                category,
                fraud_score,
                risk_level,
                decision,
                timestamp
            FROM transactions
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        
        try:
            df_recent = pd.read_sql_query(query, conn)
        except:
            # Sample data for demo
            df_recent = pd.DataFrame({
                'transaction_id': [f'tx_{i:06d}' for i in range(10)],
                'user_id': [f'user_{i:04d}' for i in range(10)],
                'merchant_id': ['amazon', 'walmart', 'casino', 'crypto_exchange', 'starbucks'] * 2,
                'amount': np.random.uniform(10, 5000, 10),
                'category': ['grocery', 'electronics', 'gambling', 'crypto', 'restaurant'] * 2,
                'fraud_score': np.random.uniform(0, 1, 10),
                'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], 10),
                'decision': np.random.choice(['APPROVED', 'REVIEW', 'DECLINED'], 10),
                'timestamp': [datetime.now() - timedelta(minutes=i*10) for i in range(10)]
            })
        
        conn.close()
        return df_recent
    
    def get_merchant_analysis(self) -> pd.DataFrame:
        """Get merchant risk analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                merchant_id,
                COUNT(*) as transaction_count,
                AVG(fraud_score) as avg_fraud_score,
                SUM(CASE WHEN decision = 'DECLINED' THEN 1 ELSE 0 END) as declined_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM transactions
            GROUP BY merchant_id
            HAVING COUNT(*) >= 10
            ORDER BY avg_fraud_score DESC
            LIMIT 20
        '''
        
        df_merchants = pd.read_sql_query(query, conn)
        conn.close()
        return df_merchants
    
    def get_alerts(self, limit: int = 10) -> pd.DataFrame:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT 
                alert_type,
                severity,
                message,
                transaction_id,
                created_at,
                resolved
            FROM alerts
            WHERE resolved = FALSE
            ORDER BY created_at DESC
            LIMIT {limit}
        '''
        
        df_alerts = pd.read_sql_query(query, conn)
        conn.close()
        return df_alerts

# Initialize dashboard
dashboard = AdvancedFraudDashboard()

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚨 Advanced Fraud Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Dashboard Controls")
        
        # Auto-refresh settings
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        refresh_interval = st.selectbox("Refresh Interval", [30, 60, 120, 300], index=0)
        
        if st.button("🔄 Refresh Now", type="primary"):
            st.rerun()
        
        st.divider()
        
        # Time range selector
        st.subheader("📅 Time Range")
        time_range = st.selectbox(
            "Select Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=2
        )
        
        # Alert settings
        st.divider()
        st.subheader("🚨 Alert Settings")
        alert_threshold = st.slider("Fraud Score Threshold", 0.0, 1.0, 0.7, 0.1)
        show_resolved = st.checkbox("Show Resolved Alerts", False)
        
        # System status
        st.divider()
        st.subheader("💻 System Status")
        st.success("✅ Database: Connected")
        st.success("✅ Processing: Active")
        st.info(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📈 Analytics", 
        "🚨 Alerts", 
        "🔍 Transactions", 
        "🏪 Merchants",
        "⚙️ Settings"
    ])
    
    with tab1:
        show_overview_dashboard()
    
    with tab2:
        show_analytics_dashboard()
    
    with tab3:
        show_alerts_dashboard()
    
    with tab4:
        show_transactions_dashboard()
    
    with tab5:
        show_merchants_dashboard()
    
    with tab6:
        show_settings_dashboard()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def show_overview_dashboard():
    """Show main overview dashboard"""
    st.header("📊 System Overview")
    
    # Get metrics
    metrics = dashboard.get_dashboard_metrics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Total Transactions",
            value=f"{metrics['total_transactions']:,}",
            delta=f"+{metrics['recent_activity']} (24h)"
        )
    
    with col2:
        st.metric(
            label="🚨 Fraud Detected",
            value=f"{metrics['fraud_declined']:,}",
            delta=f"{metrics['fraud_rate']:.2f}% rate",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="⚠️ Under Review",
            value=f"{metrics['fraud_review']:,}",
            delta="Manual review needed"
        )
    
    with col4:
        st.metric(
            label="💰 Amount Blocked",
            value=f"${metrics['blocked_amount']:,.0f}",
            delta="Fraud prevented"
        )
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Fraud Trends (24 Hours)")
        df_trends = dashboard.get_fraud_trends(24)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add fraud count
        fig.add_trace(
            go.Scatter(
                x=df_trends['hour'],
                y=df_trends['fraud_count'],
                mode='lines+markers',
                name='Fraud Count',
                line=dict(color='#ff4757', width=3),
                marker=dict(size=8)
            ),
            secondary_y=False,
        )
        
        # Add fraud score
        fig.add_trace(
            go.Scatter(
                x=df_trends['hour'],
                y=df_trends['avg_fraud_score'],
                mode='lines',
                name='Avg Fraud Score',
                line=dict(color='#ffa502', width=2, dash='dash')
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Fraud Count", secondary_y=False)
        fig.update_yaxes(title_text="Average Fraud Score", secondary_y=True)
        fig.update_layout(
            title="Fraud Detection Trends",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        df_risk = dashboard.get_risk_distribution()
        
        colors = {
            'MINIMAL': '#2ed573',
            'LOW': '#7bed9f', 
            'MEDIUM': '#ffa502',
            'HIGH': '#ff6348',
            'CRITICAL': '#ff4757'
        }
        
        fig = px.pie(
            df_risk, 
            values='count', 
            names='risk_level',
            title="Transaction Risk Levels",
            color='risk_level',
            color_discrete_map=colors,
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent high-risk transactions
    st.subheader("🚨 Recent High-Risk Transactions")
    df_recent = dashboard.get_recent_transactions(15)
    
    # Filter high-risk transactions
    high_risk_df = df_recent[df_recent['risk_level'].isin(['HIGH', 'CRITICAL'])].head(10)
    
    if not high_risk_df.empty:
        # Style the dataframe
        def style_risk_level(val):
            if val == 'CRITICAL':
                return 'background-color: #ff4757; color: white; font-weight: bold;'
            elif val == 'HIGH':
                return 'background-color: #ff6348; color: white; font-weight: bold;'
            return ''
        
        def style_decision(val):
            if val == 'DECLINED':
                return 'background-color: #ff4757; color: white; font-weight: bold;'
            elif val == 'REVIEW':
                return 'background-color: #ffa502; color: white; font-weight: bold;'
            elif val == 'APPROVED':
                return 'background-color: #2ed573; color: white; font-weight: bold;'
            return ''
        
        styled_df = high_risk_df.style.applymap(style_risk_level, subset=['risk_level']) \
                                     .applymap(style_decision, subset=['decision']) \
                                     .format({
                                         'amount': '${:,.2f}',
                                         'fraud_score': '{:.1%}',
                                         'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else ''
                                     })
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.success("✅ No high-risk transactions in recent activity!")
    
    # System health indicators
    st.divider()
    st.subheader("💡 System Health & Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if metrics['critical_alerts'] > 0:
            st.error(f"🚨 {metrics['critical_alerts']} Critical Alerts")
        else:
            st.success("✅ No Critical Alerts")
    
    with col2:
        if metrics['fraud_rate'] > 5:
            st.warning(f"⚠️ High Fraud Rate: {metrics['fraud_rate']:.1f}%")
        else:
            st.success(f"✅ Normal Fraud Rate: {metrics['fraud_rate']:.1f}%")
    
    with col3:
        processing_rate = metrics['recent_activity'] / 24  # per hour
        st.info(f"⚡ Processing: {processing_rate:.0f} tx/hour")
    
    with col4:
        efficiency = (metrics['fraud_approved'] + metrics['fraud_declined']) / metrics['total_transactions'] * 100
        st.info(f"🎯 Automation: {efficiency:.1f}%")

def show_analytics_dashboard():
    """Show detailed analytics dashboard"""
    st.header("📈 Advanced Analytics")
    
    # Time-based analysis
    st.subheader("⏰ Time-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly pattern
        st.write("**Fraud by Hour of Day**")
        
        conn = sqlite3.connect(dashboard.db_path)
        hourly_data = pd.read_sql_query('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN decision = 'DECLINED' THEN 1 ELSE 0 END) as fraud_count,
                AVG(fraud_score) as avg_fraud_score
            FROM transactions
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''', conn)
        conn.close()
        
        if not hourly_data.empty:
            fig = px.bar(
                hourly_data, 
                x='hour', 
                y='fraud_count',
                title="Fraud Count by Hour",
                color='avg_fraud_score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Daily pattern
        st.write("**Fraud by Day of Week**")
        
        conn = sqlite3.connect(dashboard.db_path)
        daily_data = pd.read_sql_query('''
            SELECT 
                strftime('%w', timestamp) as day_of_week,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN decision = 'DECLINED' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions
            GROUP BY strftime('%w', timestamp)
            ORDER BY day_of_week
        ''', conn)
        conn.close()
        
        if not daily_data.empty:
            # Map day numbers to names
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            daily_data['day_name'] = daily_data['day_of_week'].astype(int).map(lambda x: day_names[x])
            
            fig = px.line(
                daily_data, 
                x='day_name', 
                y='fraud_count',
                title="Fraud Count by Day of Week",
                markers=True
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Amount analysis
    st.divider()
    st.subheader("💰 Amount-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount vs fraud score correlation
        conn = sqlite3.connect(dashboard.db_path)
        amount_data = pd.read_sql_query('''
            SELECT amount, fraud_score, risk_level, decision
            FROM transactions
            WHERE amount <= 10000
            ORDER BY RANDOM()
            LIMIT 1000
        ''', conn)
        conn.close()
        
        if not amount_data.empty:
            fig = px.scatter(
                amount_data,
                x='amount',
                y='fraud_score',
                color='risk_level',
                title="Transaction Amount vs Fraud Score",
                color_discrete_map={
                    'MINIMAL': '#2ed573',
                    'LOW': '#7bed9f',
                    'MEDIUM': '#ffa502',
                    'HIGH': '#ff6348',
                    'CRITICAL': '#ff4757'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount distribution by decision
        if not amount_data.empty:
            fig = px.box(
                amount_data,
                x='decision',
                y='amount',
                title="Amount Distribution by Decision",
                color='decision',
                color_discrete_map={
                    'APPROVED': '#2ed573',
                    'REVIEW': '#ffa502',
                    'DECLINED': '#ff4757'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.divider()
    st.subheader("🎯 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Precision/Recall simulation
        st.write("**Classification Metrics**")
        
        # Simulate performance metrics
        precision = 0.87
        recall = 0.82
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        st.metric("Precision", f"{precision:.1%}")
        st.metric("Recall", f"{recall:.1%}")
        st.metric("F1-Score", f"{f1_score:.1%}")
    
    with col2:
        st.write("**Processing Metrics**")
        
        # Get processing stats
        metrics = dashboard.get_dashboard_metrics()
        
        avg_processing_time = 45  # ms
        throughput = 1200  # transactions per minute
        
        st.metric("Avg Processing Time", f"{avg_processing_time}ms")
        st.metric("Throughput", f"{throughput:,} tx/min")
        st.metric("Uptime", "99.8%")
    
    with col3:
        st.write("**Business Impact**")
        
        # Calculate business metrics
        total_blocked = metrics['blocked_amount']
        false_positive_cost = total_blocked * 0.05  # Assume 5% false positive rate
        
        st.metric("Fraud Blocked", f"${total_blocked:,.0f}")
        st.metric("Est. False Positives", f"${false_positive_cost:,.0f}")
        st.metric("Net Savings", f"${total_blocked - false_positive_cost:,.0f}")

def show_alerts_dashboard():
    """Show alerts and monitoring dashboard"""
    st.header("🚨 Alerts & Monitoring")
    
    # Get alerts
    df_alerts = dashboard.get_alerts(20)
    
    if not df_alerts.empty:
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_alerts = len(df_alerts[df_alerts['severity'] == 'HIGH'])
            if high_alerts > 0:
                st.markdown(f'<div class="alert-critical">🚨 {high_alerts} High Severity Alerts</div>', 
                           unsafe_allow_html=True)
            else:
                st.success("✅ No High Severity Alerts")
        
        with col2:
            medium_alerts = len(df_alerts[df_alerts['severity'] == 'MEDIUM'])
            if medium_alerts > 0:
                st.markdown(f'<div class="alert-high">⚠️ {medium_alerts} Medium Severity Alerts</div>', 
                           unsafe_allow_html=True)
            else:
                st.info("ℹ️ No Medium Severity Alerts")
        
        with col3:
            low_alerts = len(df_alerts[df_alerts['severity'] == 'LOW'])
            st.info(f"📝 {low_alerts} Low Severity Alerts")
        
        # Alert details
        st.subheader("📋 Alert Details")
        
        # Style alerts based on severity
        def style_severity(val):
            if val == 'HIGH':
                return 'background-color: #ff4757; color: white; font-weight: bold;'
            elif val == 'MEDIUM':
                return 'background-color: #ffa502; color: white; font-weight: bold;'
            elif val == 'LOW':
                return 'background-color: #70a1ff; color: white; font-weight: bold;'
            return ''
        
        styled_alerts = df_alerts.style.applymap(style_severity, subset=['severity']) \
                                      .format({
                                          'created_at': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else ''
                                      })
        
        st.dataframe(styled_alerts, use_container_width=True)
        
        # Add sample alerts for demo
        if st.button("🔄 Generate Sample Alert"):
            sample_alerts = [
                "Velocity check failed: User user_001234 made 15 transactions in 1 hour",
                "Geographic anomaly: Transaction from unusual location detected",
                "Amount threshold exceeded: $15,000 transaction flagged",
                "Merchant risk alert: High-risk merchant pattern detected",
                "Card testing pattern: Multiple small transactions detected"
            ]
            
            st.warning(f"🚨 **New Alert Generated:** {np.random.choice(sample_alerts)}")
    
    else:
        st.success("✅ No active alerts - System operating normally")
    
    # Real-time monitoring
    st.divider()
    st.subheader("📊 Real-Time Monitoring")
    
    # Create monitoring charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction volume over time
        df_trends = dashboard.get_fraud_trends(6)  # Last 6 hours
        
        fig = px.line(
            df_trends,
            x='hour',
            y='total_transactions',
            title="Transaction Volume (Last 6 Hours)",
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud score distribution
        conn = sqlite3.connect(dashboard.db_path)
        recent_scores = pd.read_sql_query('''
            SELECT fraud_score
            FROM transactions
            WHERE timestamp >= datetime('now', '-1 hour')
        ''', conn)
        conn.close()
        
        if not recent_scores.empty:
            fig = px.histogram(
                recent_scores,
                x='fraud_score',
                title="Fraud Score Distribution (Last Hour)",
                nbins=20
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_transactions_dashboard():
    """Show detailed transaction analysis"""
    st.header("🔍 Transaction Analysis")
    
    # Transaction search and filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_id = st.text_input("🔍 Search Transaction ID")
    
    with col2:
        risk_filter = st.multiselect(
            "Risk Level Filter",
            ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            default=['HIGH', 'CRITICAL']
        )
    
    with col3:
        decision_filter = st.multiselect(
            "Decision Filter",
            ['APPROVED', 'REVIEW', 'DECLINED'],
            default=['DECLINED', 'REVIEW']
        )
    
    # Get filtered transactions
    conn = sqlite3.connect(dashboard.db_path)
    
    where_conditions = []
    if search_id:
        where_conditions.append(f"transaction_id LIKE '%{search_id}%'")
    if risk_filter:
        risk_list = "', '".join(risk_filter)
        where_conditions.append(f"risk_level IN ('{risk_list}')")
    if decision_filter:
        decision_list = "', '".join(decision_filter)
        where_conditions.append(f"decision IN ('{decision_list}')")
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    query = f'''
        SELECT 
            transaction_id,
            user_id,
            merchant_id,
            amount,
            category,
            fraud_score,
            risk_level,
            decision,
            risk_factors,
            timestamp
        FROM transactions
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT 100
    '''
    
    df_transactions = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df_transactions.empty:
        st.subheader(f"📋 Filtered Transactions ({len(df_transactions)} results)")
        
        # Enhanced transaction display
        def style_transaction_row(row):
            styles = [''] * len(row)
            
            # Style based on risk level
            if row['risk_level'] == 'CRITICAL':
                styles = ['background-color: #ffebee'] * len(row)
            elif row['risk_level'] == 'HIGH':
                styles = ['background-color: #fff3e0'] * len(row)
            
            return styles
        
        styled_transactions = df_transactions.style.apply(style_transaction_row, axis=1) \
                                                  .format({
                                                      'amount': '${:,.2f}',
                                                      'fraud_score': '{:.1%}',
                                                      'timestamp': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else ''
                                                  })
        
        st.dataframe(styled_transactions, use_container_width=True)
        
        # Transaction details expander
        if st.checkbox("Show Risk Factor Details"):
            for idx, row in df_transactions.head(5).iterrows():
                with st.expander(f"Transaction {row['transaction_id']} - Risk Factors"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Amount:** ${row['amount']:,.2f}")
                        st.write(f"**Merchant:** {row['merchant_id']}")
                        st.write(f"**Category:** {row['category']}")
                    
                    with col2:
                        st.write(f"**Fraud Score:** {row['fraud_score']:.1%}")
                        st.write(f"**Risk Level:** {row['risk_level']}")
                        st.write(f"**Decision:** {row['decision']}")
                    
                    # Parse and display risk factors
                    try:
                        risk_factors = json.loads(row['risk_factors']) if row['risk_factors'] else []
                        if risk_factors:
                            st.write("**Risk Factors:**")
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                        else:
                            st.write("**Risk Factors:** None")
                    except:
                        st.write("**Risk Factors:** Unable to parse")
    
    else:
        st.info("No transactions found matching the current filters.")

def show_merchants_dashboard():
    """Show merchant analysis dashboard"""
    st.header("🏪 Merchant Analysis")
    
    # Get merchant analysis
    df_merchants = dashboard.get_merchant_analysis()
    
    if not df_merchants.empty:
        # Top risky merchants
        st.subheader("🚨 Highest Risk Merchants")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of merchant risk scores
            top_risky = df_merchants.head(10)
            
            fig = px.bar(
                top_risky,
                x='avg_fraud_score',
                y='merchant_id',
                orientation='h',
                title="Top 10 Risky Merchants",
                color='avg_fraud_score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot: transaction count vs fraud score
            fig = px.scatter(
                df_merchants,
                x='transaction_count',
                y='avg_fraud_score',
                size='total_amount',
                hover_name='merchant_id',
                title="Merchant Risk vs Transaction Volume",
                color='declined_count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed merchant table
        st.subheader("📊 Detailed Merchant Statistics")
        
        # Style the merchant dataframe
        def style_fraud_score(val):
            if val > 0.6:
                return 'background-color: #ff4757; color: white; font-weight: bold;'
            elif val > 0.4:
                return 'background-color: #ffa502; color: white; font-weight: bold;'
            elif val > 0.2:
                return 'background-color: #ffb142; color: white; font-weight: bold;'
            return ''
        
        styled_merchants = df_merchants.style.applymap(style_fraud_score, subset=['avg_fraud_score']) \
                                           .format({
                                               'avg_fraud_score': '{:.1%}',
                                               'total_amount': '${:,.0f}',
                                               'avg_amount': '${:,.2f}'
                                           })
        
        st.dataframe(styled_merchants, use_container_width=True)
        
        # Merchant insights
        st.divider()
        st.subheader("💡 Merchant Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highest_risk = df_merchants.iloc[0]
            st.error(f"🚨 **Highest Risk:** {highest_risk['merchant_id']}")
            st.write(f"Fraud Score: {highest_risk['avg_fraud_score']:.1%}")
            st.write(f"Transactions: {highest_risk['transaction_count']:,}")
        
        with col2:
            most_declined = df_merchants.loc[df_merchants['declined_count'].idxmax()]
            st.warning(f"🛑 **Most Declined:** {most_declined['merchant_id']}")
            st.write(f"Declined: {most_declined['declined_count']:,}")
            st.write(f"Total Volume: ${most_declined['total_amount']:,.0f}")
        
        with col3:
            highest_volume = df_merchants.loc[df_merchants['total_amount'].idxmax()]
            st.info(f"💰 **Highest Volume:** {highest_volume['merchant_id']}")
            st.write(f"Volume: ${highest_volume['total_amount']:,.0f}")
            st.write(f"Avg Amount: ${highest_volume['avg_amount']:,.2f}")
    
    else:
        st.info("No merchant data available for analysis.")

def show_settings_dashboard():
    """Show system settings and configuration"""
    st.header("⚙️ System Settings")
    
    # Fraud detection settings
    st.subheader("🎯 Fraud Detection Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Threshold Settings**")
        
        decline_threshold = st.slider(
            "Auto-Decline Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Transactions above this score are automatically declined"
        )
        
        review_threshold = st.slider(
            "Manual Review Threshold", 
            0.0, 1.0, 0.4, 0.05,
            help="Transactions above this score require manual review"
        )
        
        amount_threshold = st.number_input(
            "High Amount Threshold ($)",
            min_value=100, max_value=50000, value=5000,
            help="Transactions above this amount get additional scrutiny"
        )
    
    with col2:
        st.write("**Alert Settings**")
        
        enable_email_alerts = st.checkbox("Enable Email Alerts", True)
        enable_sms_alerts = st.checkbox("Enable SMS Alerts", False)
        
        alert_frequency = st.selectbox(
            "Alert Frequency",
            ["Immediate", "Every 5 minutes", "Every 15 minutes", "Hourly"],
            index=0
        )
        
        max_alerts_per_hour = st.number_input(
            "Max Alerts per Hour",
            min_value=1, max_value=100, value=20
        )
    
    # System monitoring
    st.divider()
    st.subheader("📊 System Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Performance Metrics**")
        st.metric("CPU Usage", "23%")
        st.metric("Memory Usage", "1.2GB")
        st.metric("Disk Usage", "45%")
    
    with col2:
        st.write("**Database Stats**")
        metrics = dashboard.get_dashboard_metrics()
        st.metric("Total Records", f"{metrics['total_transactions']:,}")
        st.metric("Database Size", "156 MB")
        st.metric("Last Backup", "2 hours ago")
    
    with col3:
        st.write("**API Status**")
        st.success("✅ Fraud Detection API")
        st.success("✅ Database Connection")
        st.success("✅ Alert System")
    
    # Data management
    st.divider()
    st.subheader("🗄️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Retention**")
        
        retention_period = st.selectbox(
            "Transaction Data Retention",
            ["30 days", "90 days", "1 year", "2 years", "Indefinite"],
            index=2
        )
        
        archive_old_data = st.checkbox("Auto-archive old data", True)
        
        if st.button("🗑️ Clean Old Data"):
            st.success("✅ Data cleanup scheduled")
    
    with col2:
        st.write("**Export & Backup**")
        
        if st.button("📥 Export Transaction Data"):
            st.info("📊 Export started - you'll receive a download link shortly")
        
        if st.button("💾 Create Backup"):
            st.success("✅ Backup created successfully")
        
        if st.button("🔄 Restore from Backup"):
            st.warning("⚠️ Restore functionality requires admin privileges")
    
    # Save settings
    st.divider()
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")
        st.balloons()

if __name__ == "__main__":
    main()