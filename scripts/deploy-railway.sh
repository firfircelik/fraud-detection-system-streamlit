#!/bin/bash
# Railway deployment için hazırlık

echo "🚂 Railway.app deployment hazırlığı..."

# Railway için Dockerfile kontrol
if [ -f "docker/Dockerfile.streamlit" ]; then
    echo "✅ Dockerfile bulundu"
else
    echo "❌ Dockerfile bulunamadı"
    exit 1
fi

# Railway için port ayarı
export PORT=8501

echo "🔧 Railway deployment komutları:"
echo "1. https://railway.app adresine git"
echo "2. GitHub ile giriş yap"
echo "3. 'New Project' > 'Deploy from GitHub repo'"
echo "4. Bu repo'yu seç: fraud-detection-system-streamlit"
echo "5. Build Command: docker build -f docker/Dockerfile.streamlit -t app ."
echo "6. Start Command: streamlit run app/main.py --server.port \$PORT"
echo ""
echo "🌍 Deployment sonrası URL: [railway-subdomain].railway.app"
