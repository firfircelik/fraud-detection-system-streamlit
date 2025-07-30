#!/bin/bash

# 🧹 Scala Cleanup Script
# Scala ile ilgili tüm dosyaları temizler

echo "🧹 Scala Cleanup - Removing Scala-related files..."
echo "================================================="

# Scala source files
echo "🗑️ Removing Scala source files..."
rm -rf src/main/scala/
rm -rf src/test/scala/
rm -rf src/backup/scala/

# SBT build files
echo "🗑️ Removing SBT build files..."
rm -f build.sbt
rm -rf project/

# Scala build outputs
echo "🗑️ Removing Scala build outputs..."
rm -rf target/

# SBT cache and temp files
echo "🗑️ Removing SBT cache..."
rm -rf ~/.sbt/boot/
rm -rf .bsp/

# IntelliJ IDEA Scala files
echo "🗑️ Removing IDE files..."
rm -rf .idea/
rm -f *.iml

# Metals (Scala Language Server) files
echo "🗑️ Removing Metals files..."
rm -rf .metals/
rm -rf .bloop/

echo ""
echo "✅ Scala cleanup completed!"
echo "📊 Project is now pure Python/Streamlit"
echo ""
echo "📁 Remaining structure:"
ls -la | grep -E "(streamlit|csv|python|requirements|docker|\.py$|\.sh$|\.yml$|\.md$)"
