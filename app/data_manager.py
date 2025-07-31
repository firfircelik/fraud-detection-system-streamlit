#!/usr/bin/env python3
"""
🚨 Central Data Manager for Fraud Detection System
Merkezi veri yönetimi sistemi - CSV verilerini tüm modüller arasında paylaşır
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import json

class CentralDataManager:
    """Central data management system"""
    
    def __init__(self):
        self.session_keys = {
            'loaded_data': 'central_loaded_data',
            'processed_data': 'central_processed_data',
            'metadata': 'central_metadata',
            'analysis_cache': 'central_analysis_cache'
        }
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        for key in self.session_keys.values():
            if key not in st.session_state:
                st.session_state[key] = None
    
    def load_csv_data(self, df: pd.DataFrame, filename: str = "uploaded_data.csv", 
                     source: str = "upload") -> bool:
        """Load CSV data into central storage"""
        try:
            if df is None or df.empty:
                return False
            
            # Store raw data
            st.session_state[self.session_keys['loaded_data']] = df.copy()
            
            # Create metadata
            metadata = {
                'filename': filename,
                'source': source,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'loaded_at': datetime.now().isoformat(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                'has_fraud_column': any(col in df.columns for col in ['is_fraud', 'fraud', 'Class', 'isFraud']),
                'has_amount_column': 'amount' in df.columns or 'Amount' in df.columns,
                'has_timestamp_column': any('time' in col.lower() or 'date' in col.lower() for col in df.columns),
                'has_user_column': any('user' in col.lower() or 'customer' in col.lower() for col in df.columns),
                'has_merchant_column': any('merchant' in col.lower() for col in df.columns),
                'has_geo_columns': any(col in df.columns for col in ['lat', 'lon', 'latitude', 'longitude'])
            }
            
            st.session_state[self.session_keys['metadata']] = metadata
            
            # Clear processed data and cache when new data is loaded
            st.session_state[self.session_keys['processed_data']] = None
            st.session_state[self.session_keys['analysis_cache']] = {}
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Get raw loaded data"""
        return st.session_state.get(self.session_keys['loaded_data'])
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed fraud detection data"""
        return st.session_state.get(self.session_keys['processed_data'])
    
    def set_processed_data(self, df: pd.DataFrame):
        """Set processed fraud detection data"""
        st.session_state[self.session_keys['processed_data']] = df.copy()
    
    def get_metadata(self) -> Optional[Dict]:
        """Get data metadata"""
        return st.session_state.get(self.session_keys['metadata'])
    
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self.get_raw_data() is not None
    
    def has_processed_data(self) -> bool:
        """Check if processed data is available"""
        return self.get_processed_data() is not None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        metadata = self.get_metadata()
        raw_data = self.get_raw_data()
        processed_data = self.get_processed_data()
        
        if not metadata or raw_data is None:
            return {}
        
        summary = {
            'basic_info': {
                'filename': metadata['filename'],
                'source': metadata['source'],
                'rows': metadata['rows'],
                'columns': metadata['columns'],
                'memory_mb': metadata['memory_usage'],
                'loaded_at': metadata['loaded_at']
            },
            'capabilities': {
                'fraud_analysis': metadata['has_fraud_column'],
                'financial_analysis': metadata['has_amount_column'],
                'temporal_analysis': metadata['has_timestamp_column'],
                'behavioral_analysis': metadata['has_user_column'],
                'merchant_analysis': metadata['has_merchant_column'],
                'geographic_analysis': metadata['has_geo_columns']
            },
            'processing_status': {
                'raw_data_available': True,
                'processed_data_available': processed_data is not None,
                'fraud_detection_ready': processed_data is not None
            }
        }
        
        # Add fraud statistics if processed data is available
        if processed_data is not None:
            if 'decision' in processed_data.columns:
                decisions = processed_data['decision'].value_counts().to_dict()
                summary['fraud_stats'] = {
                    'total_transactions': len(processed_data),
                    'approved': decisions.get('APPROVED', 0),
                    'declined': decisions.get('DECLINED', 0),
                    'review': decisions.get('REVIEW', 0),
                    'fraud_rate': (decisions.get('DECLINED', 0) / len(processed_data)) * 100
                }
            
            if 'risk_level' in processed_data.columns:
                risk_levels = processed_data['risk_level'].value_counts().to_dict()
                summary['risk_distribution'] = risk_levels
        
        return summary
    
    def clear_data(self):
        """Clear all loaded data"""
        for key in self.session_keys.values():
            st.session_state[key] = None
    
    def cache_analysis_result(self, analysis_type: str, result: Any):
        """Cache analysis results"""
        if self.session_keys['analysis_cache'] not in st.session_state:
            st.session_state[self.session_keys['analysis_cache']] = {}
        
        st.session_state[self.session_keys['analysis_cache']][analysis_type] = {
            'result': result,
            'cached_at': datetime.now().isoformat()
        }
    
    def get_cached_analysis(self, analysis_type: str) -> Optional[Any]:
        """Get cached analysis result"""
        cache = st.session_state.get(self.session_keys['analysis_cache'], {})
        cached_item = cache.get(analysis_type)
        
        if cached_item:
            # Check if cache is still valid (less than 1 hour old)
            cached_time = datetime.fromisoformat(cached_item['cached_at'])
            if (datetime.now() - cached_time).seconds < 3600:
                return cached_item['result']
        
        return None

# Global instance
data_manager = CentralDataManager()

def show_data_status_widget():
    """Show current data status widget"""
    if not data_manager.has_data():
        st.info("📂 No data loaded. Please upload a CSV file in the Data Explorer tab.")
        return
    
    summary = data_manager.get_data_summary()
    
    if summary:
        with st.expander("📊 Current Dataset Info", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📁 File", summary['basic_info']['filename'])
                st.metric("📊 Rows", f"{summary['basic_info']['rows']:,}")
            
            with col2:
                st.metric("📋 Columns", summary['basic_info']['columns'])
                st.metric("💾 Size", f"{summary['basic_info']['memory_mb']:.1f} MB")
            
            with col3:
                if summary['processing_status']['processed_data_available']:
                    st.success("✅ Fraud Analysis Ready")
                    if 'fraud_stats' in summary:
                        fraud_rate = summary['fraud_stats']['fraud_rate']
                        st.metric("🚨 Fraud Rate", f"{fraud_rate:.2f}%")
                else:
                    st.warning("⚠️ Process data first")
            
            # Capabilities
            st.write("**Available Analysis Types:**")
            capabilities = summary['capabilities']
            
            cap_cols = st.columns(3)
            with cap_cols[0]:
                st.write("✅ Fraud Analysis" if capabilities['fraud_analysis'] else "❌ Fraud Analysis")
                st.write("✅ Financial Analysis" if capabilities['financial_analysis'] else "❌ Financial Analysis")
            
            with cap_cols[1]:
                st.write("✅ Temporal Analysis" if capabilities['temporal_analysis'] else "❌ Temporal Analysis")
                st.write("✅ Behavioral Analysis" if capabilities['behavioral_analysis'] else "❌ Behavioral Analysis")
            
            with cap_cols[2]:
                st.write("✅ Merchant Analysis" if capabilities['merchant_analysis'] else "❌ Merchant Analysis")
                st.write("✅ Geographic Analysis" if capabilities['geographic_analysis'] else "❌ Geographic Analysis")

def require_data(func):
    """Decorator to require data before running function"""
    def wrapper(*args, **kwargs):
        if not data_manager.has_data():
            st.warning("⚠️ Please load data first using the Data Explorer tab.")
            return None
        return func(*args, **kwargs)
    return wrapper

def require_processed_data(func):
    """Decorator to require processed data before running function"""
    def wrapper(*args, **kwargs):
        if not data_manager.has_processed_data():
            st.warning("⚠️ Please process data first using the CSV Processor tab.")
            return None
        return func(*args, **kwargs)
    return wrapper