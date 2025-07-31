#!/usr/bin/env python3
"""
🚨 Advanced Data Loader for Fraud Detection
Büyük veri dosyalarını işlemek için gelişmiş veri yükleyici
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataLoader:
    """Advanced data loader for large datasets"""
    
    def __init__(self):
        self.data_dir = "data"
        self.massive_dir = os.path.join(self.data_dir, "massive")
        self.supported_formats = ['.csv', '.json', '.jsonl']
    
    def get_available_datasets(self) -> List[Dict]:
        """Get list of available datasets"""
        datasets = []
        
        # Check regular data directory
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if any(filename.endswith(ext) for ext in self.supported_formats):
                    filepath = os.path.join(self.data_dir, filename)
                    size = os.path.getsize(filepath)
                    datasets.append({
                        'name': filename,
                        'path': filepath,
                        'size_bytes': size,
                        'size_mb': round(size / (1024 * 1024), 2),
                        'location': 'data',
                        'type': 'sample'
                    })
        
        # Check massive data directory
        if os.path.exists(self.massive_dir):
            for filename in os.listdir(self.massive_dir):
                if any(filename.endswith(ext) for ext in self.supported_formats):
                    filepath = os.path.join(self.massive_dir, filename)
                    size = os.path.getsize(filepath)
                    datasets.append({
                        'name': filename,
                        'path': filepath,
                        'size_bytes': size,
                        'size_mb': round(size / (1024 * 1024), 2),
                        'location': 'massive',
                        'type': 'large'
                    })
        
        return sorted(datasets, key=lambda x: x['size_bytes'], reverse=True)
    
    def show_dataset_selector(self) -> Optional[Dict]:
        """Show dataset selector interface"""
        st.subheader("📊 Available Datasets")
        
        datasets = self.get_available_datasets()
        
        if not datasets:
            st.warning("⚠️ No datasets found in data directories")
            return None
        
        # Create dataset selection interface
        dataset_options = []
        for dataset in datasets:
            size_str = f"{dataset['size_mb']:.1f} MB" if dataset['size_mb'] < 1000 else f"{dataset['size_mb']/1000:.1f} GB"
            option_text = f"{dataset['name']} ({size_str}) - {dataset['location']}"
            dataset_options.append(option_text)
        
        selected_option = st.selectbox(
            "Select Dataset",
            options=dataset_options,
            help="Choose a dataset to analyze"
        )
        
        if selected_option:
            selected_index = dataset_options.index(selected_option)
            selected_dataset = datasets[selected_index]
            
            # Show dataset info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📁 File Size", f"{selected_dataset['size_mb']:.1f} MB")
            
            with col2:
                st.metric("📍 Location", selected_dataset['location'])
            
            with col3:
                st.metric("📊 Type", selected_dataset['type'])
            
            with col4:
                # Estimate row count
                estimated_rows = self.estimate_row_count(selected_dataset)
                st.metric("📈 Est. Rows", f"{estimated_rows:,}")
            
            return selected_dataset
        
        return None
    
    def estimate_row_count(self, dataset: Dict) -> int:
        """Estimate number of rows in dataset"""
        try:
            if dataset['name'].endswith('.csv'):
                # Quick estimation based on file size
                # Assume average row size of 200 bytes
                estimated_rows = dataset['size_bytes'] // 200
                return max(1, estimated_rows)
            elif dataset['name'].endswith('.json'):
                # JSON files are typically smaller in row count
                estimated_rows = dataset['size_bytes'] // 500
                return max(1, estimated_rows)
            elif dataset['name'].endswith('.jsonl'):
                # JSONL files, estimate based on average line size
                estimated_rows = dataset['size_bytes'] // 300
                return max(1, estimated_rows)
        except:
            pass
        
        return 1000  # Default estimate
    
    def load_dataset_sample(self, dataset: Dict, sample_size: int = 1000) -> pd.DataFrame:
        """Load a sample from the dataset"""
        filepath = dataset['path']
        
        try:
            if dataset['name'].endswith('.csv'):
                return self.load_csv_sample(filepath, sample_size)
            elif dataset['name'].endswith('.json'):
                return self.load_json_sample(filepath, sample_size)
            elif dataset['name'].endswith('.jsonl'):
                return self.load_jsonl_sample(filepath, sample_size)
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return pd.DataFrame()
        
        return pd.DataFrame()
    
    def load_csv_sample(self, filepath: str, sample_size: int) -> pd.DataFrame:
        """Load CSV sample with progress tracking"""
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        if file_size_mb > 100:
            st.info(f"📊 Loading large file ({file_size_mb:.1f} MB) - this may take a moment...")
            
            # For very large files, use chunked reading
            chunk_size = 10000
            chunks_to_read = max(1, sample_size // chunk_size)
            
            chunks = []
            rows_read = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
                    chunks.append(chunk)
                    rows_read += len(chunk)
                    
                    progress = min(1.0, (i + 1) / chunks_to_read)
                    progress_bar.progress(progress)
                    status_text.text(f"Loading... {rows_read:,} rows read")
                    
                    if rows_read >= sample_size:
                        break
                
                progress_bar.empty()
                status_text.empty()
                
                df = pd.concat(chunks, ignore_index=True)
                return df.head(sample_size)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                raise e
        else:
            # For smaller files, load normally
            return pd.read_csv(filepath).head(sample_size)
    
    def load_json_sample(self, filepath: str, sample_size: int) -> pd.DataFrame:
        """Load JSON sample"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'transactions' in data:
            # Handle nested JSON structure
            transactions = data['transactions'][:sample_size]
            return pd.json_normalize(transactions)
        elif isinstance(data, list):
            # Handle list of records
            return pd.DataFrame(data[:sample_size])
        else:
            st.error("❌ Unsupported JSON structure")
            return pd.DataFrame()
    
    def load_jsonl_sample(self, filepath: str, sample_size: int) -> pd.DataFrame:
        """Load JSONL sample"""
        records = []
        
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        return pd.DataFrame(records)
    
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset structure and provide insights"""
        analysis = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            },
            'columns': {},
            'data_quality': {},
            'fraud_indicators': {}
        }
        
        # Analyze each column
        for col in df.columns:
            col_analysis = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Add type-specific analysis
            if df[col].dtype in ['int64', 'float64']:
                col_analysis.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                })
            elif df[col].dtype == 'object':
                col_analysis.update({
                    'top_values': df[col].value_counts().head(5).to_dict()
                })
            
            analysis['columns'][col] = col_analysis
        
        # Data quality assessment
        analysis['data_quality'] = {
            'completeness': (df.count().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Fraud-related indicators
        fraud_columns = ['is_fraud', 'fraud', 'Class', 'isFraud']
        fraud_col = None
        for col in fraud_columns:
            if col in df.columns:
                fraud_col = col
                break
        
        if fraud_col:
            fraud_rate = (df[fraud_col].sum() / len(df)) * 100
            analysis['fraud_indicators'] = {
                'fraud_column': fraud_col,
                'fraud_rate': fraud_rate,
                'fraud_count': df[fraud_col].sum(),
                'legitimate_count': len(df) - df[fraud_col].sum()
            }
        
        return analysis
    
    def show_dataset_analysis(self, df: pd.DataFrame, analysis: Dict):
        """Show comprehensive dataset analysis"""
        st.subheader("🔍 Dataset Analysis")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", f"{analysis['basic_info']['rows']:,}")
        
        with col2:
            st.metric("📋 Columns", analysis['basic_info']['columns'])
        
        with col3:
            st.metric("💾 Memory", f"{analysis['basic_info']['memory_usage']:.1f} MB")
        
        with col4:
            st.metric("✅ Completeness", f"{analysis['data_quality']['completeness']:.1f}%")
        
        # Fraud indicators
        if analysis['fraud_indicators']:
            st.divider()
            st.subheader("🚨 Fraud Indicators")
            
            fraud_info = analysis['fraud_indicators']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Fraud Rate", f"{fraud_info['fraud_rate']:.2f}%")
            
            with col2:
                st.metric("🚨 Fraud Count", f"{fraud_info['fraud_count']:,}")
            
            with col3:
                st.metric("✅ Legitimate Count", f"{fraud_info['legitimate_count']:,}")
        
        # Column analysis
        st.divider()
        st.subheader("📋 Column Analysis")
        
        # Create column summary table
        column_summary = []
        for col, info in analysis['columns'].items():
            summary_row = {
                'Column': col,
                'Type': info['dtype'],
                'Non-Null': f"{info['non_null_count']:,}",
                'Null %': f"{info['null_percentage']:.1f}%",
                'Unique': f"{info['unique_count']:,}",
                'Unique %': f"{info['unique_percentage']:.1f}%"
            }
            column_summary.append(summary_row)
        
        column_df = pd.DataFrame(column_summary)
        st.dataframe(column_df, use_container_width=True)
        
        # Data quality issues
        st.divider()
        st.subheader("⚠️ Data Quality Assessment")
        
        quality_issues = []
        
        # Check for high null percentages
        for col, info in analysis['columns'].items():
            if info['null_percentage'] > 20:
                quality_issues.append(f"❌ **{col}**: High null percentage ({info['null_percentage']:.1f}%)")
        
        # Check for duplicate rows
        if analysis['data_quality']['duplicate_percentage'] > 1:
            quality_issues.append(f"❌ **Duplicates**: {analysis['data_quality']['duplicate_percentage']:.1f}% duplicate rows")
        
        # Check for low cardinality in potential ID columns
        for col, info in analysis['columns'].items():
            if 'id' in col.lower() and info['unique_percentage'] < 90:
                quality_issues.append(f"⚠️ **{col}**: Low uniqueness for ID column ({info['unique_percentage']:.1f}%)")
        
        if quality_issues:
            for issue in quality_issues:
                st.warning(issue)
        else:
            st.success("✅ No major data quality issues detected!")
        
        # Recommendations
        st.divider()
        st.subheader("💡 Analysis Recommendations")
        
        recommendations = []
        
        # Check for timestamp columns
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            recommendations.append(f"⏰ **Temporal Analysis**: Use {', '.join(timestamp_cols)} for time-based fraud patterns")
        
        # Check for geographic columns
        geo_cols = [col for col in df.columns if col.lower() in ['lat', 'lon', 'latitude', 'longitude']]
        if geo_cols:
            recommendations.append(f"🌍 **Geographic Analysis**: Use {', '.join(geo_cols)} for location-based analysis")
        
        # Check for user/customer columns
        user_cols = [col for col in df.columns if any(term in col.lower() for term in ['user', 'customer', 'client'])]
        if user_cols:
            recommendations.append(f"👤 **Behavioral Analysis**: Use {', '.join(user_cols)} for user behavior patterns")
        
        # Check for merchant columns
        merchant_cols = [col for col in df.columns if 'merchant' in col.lower()]
        if merchant_cols:
            recommendations.append(f"🏪 **Merchant Analysis**: Use {', '.join(merchant_cols)} for merchant risk profiling")
        
        # Check for amount columns
        amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'price'])]
        if amount_cols:
            recommendations.append(f"💰 **Financial Analysis**: Use {', '.join(amount_cols)} for transaction amount patterns")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        
        return analysis

def show_data_loader_interface():
    """Show the data loader interface"""
    loader = AdvancedDataLoader()
    
    st.header("📊 Advanced Data Loader")
    
    # Dataset selection
    selected_dataset = loader.show_dataset_selector()
    
    if selected_dataset:
        st.divider()
        
        # Sample size selection
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size = st.number_input(
                "Sample Size",
                min_value=100,
                max_value=100000,
                value=5000,
                step=500,
                help="Number of rows to load for analysis"
            )
        
        with col2:
            load_full = st.checkbox(
                "Load Full Dataset",
                help="Load entire dataset (may be slow for large files)"
            )
        
        if st.button("🚀 Load Dataset", type="primary"):
            with st.spinner(f"Loading {selected_dataset['name']}..."):
                # Load dataset
                if load_full:
                    # For full load, we need to handle this carefully
                    if selected_dataset['size_mb'] > 500:
                        st.warning("⚠️ Large dataset detected. Loading sample instead for performance.")
                        df = loader.load_dataset_sample(selected_dataset, sample_size)
                    else:
                        df = loader.load_dataset_sample(selected_dataset, sample_size * 10)  # Load more for "full"
                else:
                    df = loader.load_dataset_sample(selected_dataset, sample_size)
                
                if not df.empty:
                    st.success(f"✅ Loaded {len(df):,} rows from {selected_dataset['name']}")
                    
                    # Analyze dataset
                    analysis = loader.analyze_dataset_structure(df)
                    
                    # Show analysis
                    loader.show_dataset_analysis(df, analysis)
                    
                    # Store in session state for further analysis
                    st.session_state['loaded_dataset'] = {
                        'data': df,
                        'analysis': analysis,
                        'metadata': selected_dataset
                    }
                    
                    # Option to proceed with fraud analysis
                    st.divider()
                    
                    if st.button("🔍 Proceed to Fraud Analysis", type="primary"):
                        st.session_state['fraud_data'] = df
                        st.success("✅ Dataset loaded for fraud analysis!")
                        st.info("💡 Go to the 'Advanced Analytics' tab to see comprehensive analysis.")
                
                else:
                    st.error("❌ Failed to load dataset")
    
    return selected_dataset