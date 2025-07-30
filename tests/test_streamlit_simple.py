#!/usr/bin/env python3
"""
Simple test for Streamlit CSV functionality
"""

import streamlit as st
import pandas as pd

st.title("🧪 Simple CSV Test")

# Test file upload
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: {len(df)} rows")
        st.dataframe(df.head())
        
        if st.button("Test Process"):
            st.success("✅ Processing works!")
            st.json({"test": "success", "rows": len(df)})
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

st.write("Debug info:")
st.write("Session state keys:", list(st.session_state.keys()))