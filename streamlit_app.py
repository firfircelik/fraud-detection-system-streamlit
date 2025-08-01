#!/usr/bin/env python3
"""
Streamlit Cloud entry point for Fraud Detection Dashboard
This file serves as the main entry point for Streamlit Cloud deployment.
It imports and runs the actual application from app/main.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
from app.main import main

if __name__ == "__main__":
    main()