#!/usr/bin/env python3
"""
Fix duplicate plotly chart keys in streamlit_app.py
"""

import re

# Read the file
with open('streamlit_app.py', 'r') as f:
    content = f.read()

# Find all plotly_chart calls and add unique keys
plotly_calls = re.findall(r'st\.plotly_chart\([^)]+\)', content)

print(f"Found {len(plotly_calls)} plotly_chart calls")

# Replace each call with a unique key
key_counter = 1
for call in plotly_calls:
    if 'key=' not in call:
        # Add unique key
        new_call = call.replace(')', f', key="plotly_chart_{key_counter}")')
        content = content.replace(call, new_call, 1)  # Replace only first occurrence
        key_counter += 1

# Write back
with open('streamlit_app.py', 'w') as f:
    f.write(content)

print(f"Fixed {key_counter-1} plotly_chart calls with unique keys")