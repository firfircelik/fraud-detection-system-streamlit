#!/usr/bin/env python3
"""
🚨 Streamlit Cloud Entry Point - Fraud Detection System
Main entry point for Streamlit Cloud deployment
"""

import sys
import os

# Add app directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the main app
from app.main import main

if __name__ == "__main__":
    main()
