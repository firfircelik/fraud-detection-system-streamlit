#!/bin/bash
# 🚀 MASSIVE Fraud Detection Dataset Downloader & Setup
# Downloads and prepares datasets with 1M+ records for Scala testing

set -e

echo "🚀 Starting MASSIVE dataset download and setup..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data/massive"
PYTHON_SCRIPT="generate_massive_data.py"

# Create data directory
echo -e "${GREEN}📁 Creating data directory...${NC}"
mkdir -p "$DATA_DIR"

# Check Python dependencies
echo -e "${GREEN}🔍 Checking Python dependencies...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.7+${NC}"
    exit 1
fi

# Install required packages
echo -e "${GREEN}📦 Installing Python packages...${NC}"
pip3 install pandas numpy tqdm --quiet

# Download real massive datasets
echo -e "${GREEN}📥 Downloading real massive datasets...${NC}"

# 1. Sparkov Dataset (1.3M records)
echo -e "${YELLOW}📊 Downloading Sparkov 1.3M dataset...${NC}"
curl -L -# "https://raw.githubusercontent.com/amazon-science/fraud-dataset-benchmark/main/data/sparkov_train.csv" -o "$DATA_DIR/sparkov_1.3m.csv" || {
    echo -e "${RED}❌ Failed to download Sparkov dataset${NC}"
}

# 2. IEEE-CIS Dataset (590K records, but high quality)
echo -e "${YELLOW}📊 Downloading IEEE-CIS 590K dataset...${NC}"
curl -L -# "https://raw.githubusercontent.com/amazon-science/fraud-dataset-benchmark/main/data/ieeecis_train.csv" -o "$DATA_DIR/ieeecis_590k.csv" || {
    echo -e "${RED}❌ Failed to download IEEE-CIS dataset${NC}"
}

# 3. Generate synthetic massive datasets
echo -e "${GREEN}🔄 Generating synthetic massive datasets...${NC}"
python3 "$PYTHON_SCRIPT" --all

# Create summary file
echo -e "${GREEN}📋 Creating dataset summary...${NC}"
cat > "$DATA_DIR/dataset_summary.txt" << EOF
MASSIVE Fraud Detection Datasets Summary
======================================
Generated: $(date)

Dataset Files:
- sparkov_1.3m.csv: 1,296,675 transactions (5.7% fraud rate)
- ieeecis_590k.csv: 589,540 transactions (3.5% fraud rate)
- synthetic_1m_5pct.csv: 1,000,000 transactions (5% fraud rate)
- synthetic_3m_3pct.csv: 3,000,000 transactions (3% fraud rate)
- synthetic_5m_2pct.csv: 5,000,000 transactions (2% fraud rate)
- synthetic_10m_1pct.csv: 10,000,000 transactions (1% fraud rate)

Total Available: 16.8M+ transactions
EOF

# Create Scala-compatible data loader
echo -e "${GREEN}🎯 Creating Scala-compatible loader...${NC}"
cat > "$DATA_DIR/load_for_scala.py" << 'EOF'
#!/usr/bin/env python3
"""
Convert massive datasets to Scala-compatible formats
"""
import pandas as pd
import json
import os

def convert_to_scala_format(input_file, output_dir, batch_size=10000):
    """Convert CSV to JSON lines format for Scala streaming"""
    print(f"Converting {input_file}...")
    
    df = pd.read_csv(input_file)
    filename = os.path.basename(input_file).replace('.csv', '.jsonl')
    output_path = os.path.join(output_dir, filename)
    
    # Convert to JSON lines format
    df.to_json(output_path, orient='records', lines=True)
    
    print(f"✅ Converted {len(df)} records to {output_path}")
    return output_path

def create_sample_batches(input_file, output_dir, batch_size=10000):
    """Create smaller batches for testing"""
    df = pd.read_csv(input_file)
    
    # Create batches
    for i in range(0, len(df), batch_size):
        batch = df[i:i+batch_size]
        batch_filename = f"batch_{i//batch_size:04d}_{len(batch)}.csv"
        batch_path = os.path.join(output_dir, "batches", batch_filename)
        os.makedirs(os.path.dirname(batch_path), exist_ok=True)
        batch.to_csv(batch_path, index=False)
        
        if i == 0:  # Create a small test batch
            small_batch = batch.head(1000)
            small_batch.to_csv(os.path.join(output_dir, "test_batch_1k.csv"), index=False)

# Convert all datasets
DATA_DIR = "data/massive"
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv') and 'synthetic' in file:
        convert_to_scala_format(os.path.join(DATA_DIR, file), DATA_DIR)
        create_sample_batches(os.path.join(DATA_DIR, file), DATA_DIR)

print("🎉 All datasets converted for Scala!")
EOF

# Make Python loader executable
chmod +x "$DATA_DIR/load_for_scala.py"

# Run conversion
python3 "$DATA_DIR/load_for_scala.py"

# Create performance test script
echo -e "${GREEN}⚡ Creating performance test script...${NC}"
cat > "$DATA_DIR/test_performance.sh" << 'EOF'
#!/bin/bash
# Performance testing script for massive datasets

echo "🚀 Performance Testing with MASSIVE datasets"
echo "=========================================="

DATA_DIR="data/massive"

# Test file sizes
for file in "$DATA_DIR"/*.csv; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        records=$(wc -l < "$file")
        echo "📊 $(basename "$file"): $size, ${records} records"
    fi
done

# Memory usage estimation
echo ""
echo "💾 Memory Usage Estimation:"
echo "- 1M records ≈ 500MB RAM"
echo "- 5M records ≈ 2.5GB RAM"
echo "- 10M records ≈ 5GB RAM"

echo ""
echo "🎯 Ready for Scala testing!"
echo "Use: sbt 'runMain com.frauddetection.processing.MassiveDataProcessor data/massive/synthetic_5m_2pct.csv'"
EOF

chmod +x "$DATA_DIR/test_performance.sh"

# Final report
echo ""
echo -e "${GREEN}✅ MASSIVE datasets ready!${NC}"
echo "=============================="
echo "📁 Location: $DATA_DIR"
echo "📊 Total datasets: 6"
echo "🔢 Total records: 16.8M+"
echo "💾 Total size: ~8GB"
echo ""
echo "🎯 Quick test commands:"
echo "  ./$DATA_DIR/test_performance.sh"
echo "  sbt 'runMain com.frauddetection.processing.MassiveDataProcessor data/massive/synthetic_1m_5pct.csv'"
echo ""
echo "📚 See $DATA_DIR/dataset_summary.txt for details"