#!/bin/bash
# Render.com deployment için hazırlık

echo "🎨 Render.com deployment hazırlığı..."

echo "🔧 Render deployment komutları:"
echo "1. https://render.com adresine git" 
echo "2. GitHub ile giriş yap"
echo "3. 'New' > 'Web Service'"
echo "4. Bu repo'yu bağla: fraud-detection-system-streamlit"
echo "5. Settings:"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: streamlit run app/main.py --server.port \$PORT --server.address 0.0.0.0"
echo "   - Environment: Python 3"
echo ""
echo "🌍 Deployment sonrası URL: [app-name].onrender.com"
