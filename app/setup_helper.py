#!/usr/bin/env python3
"""
🚨 Streamlit Configuration & Setup Helper
Streamlit uygulaması için konfigürasyon yardımcısı
"""

import streamlit as st
import os
import json
import subprocess
import sys
from pathlib import Path

def check_system_requirements():
    """Sistem gereksinimlerini kontrol et"""
    requirements = {
        'python': {'command': 'python3 --version', 'min_version': '3.8'},
        'pip': {'command': 'pip3 --version', 'required': True},
        'streamlit': {'command': 'streamlit version', 'required': True}
    }
    
    results = {}
    
    for name, req in requirements.items():
        try:
            result = subprocess.run(req['command'].split(), 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                results[name] = {'status': 'OK', 'output': result.stdout.strip()}
            else:
                results[name] = {'status': 'ERROR', 'output': result.stderr.strip()}
        except FileNotFoundError:
            results[name] = {'status': 'NOT_FOUND', 'output': f'{name} not found'}
    
    return results

def get_project_info():
    """Proje bilgilerini al"""
    project_root = Path.cwd()
    
    info = {
        'root': str(project_root),
        'files': {
            'streamlit_app.py': (project_root / 'streamlit_app.py').exists(),
            'fraud_processor.py': (project_root / 'fraud_processor.py').exists(),
            'requirements.txt': (project_root / 'requirements.txt').exists(),
            'docker-compose.yml': (project_root / 'docker-compose.yml').exists(),
        },
        'directories': {
            'data': (project_root / 'data').exists(),
            'logs': (project_root / 'logs').exists(),
            'streamlit-env': (project_root / 'streamlit-env').exists(),
        }
    }
    
    return info

def main():
    st.set_page_config(
        page_title="🚨 Streamlit Setup Helper",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Streamlit Fraud Detection - Setup Helper")
    st.write("Sistem durumu ve konfigürasyon kontrol paneli")
    
    # System Requirements Check
    st.header("🔍 Sistem Gereksinimleri")
    
    if st.button("🔄 Sistem Kontrolü Yap"):
        with st.spinner("Sistem kontrol ediliyor..."):
            results = check_system_requirements()
            
            for name, result in results.items():
                if result['status'] == 'OK':
                    st.success(f"✅ {name.upper()}: {result['output']}")
                elif result['status'] == 'ERROR':
                    st.error(f"❌ {name.upper()}: {result['output']}")
                else:
                    st.warning(f"⚠️ {name.upper()}: {result['output']}")
    
    # Project Structure
    st.header("📁 Proje Yapısı")
    
    project_info = get_project_info()
    
    st.write(f"**Proje Dizini:** `{project_info['root']}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Dosyalar")
        for file, exists in project_info['files'].items():
            if exists:
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
    
    with col2:
        st.subheader("📁 Dizinler")
        for dir_name, exists in project_info['directories'].items():
            if exists:
                st.success(f"✅ {dir_name}/")
            else:
                st.warning(f"⚠️ {dir_name}/")
    
    # Environment Variables
    st.header("🌍 Environment Variables")
    
    env_vars = {
        'STREAMLIT_SERVER_MAX_UPLOAD_SIZE': os.getenv('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', 'Not Set'),
        'FRAUD_API_URL': os.getenv('FRAUD_API_URL', 'Not Set'),
        'PYTHONPATH': os.getenv('PYTHONPATH', 'Not Set'),
    }
    
    for var, value in env_vars.items():
        if value != 'Not Set':
            st.success(f"✅ {var}: `{value}`")
        else:
            st.info(f"ℹ️ {var}: {value}")
    
    # Configuration Generator
    st.header("⚙️ Konfigürasyon Oluşturucu")
    
    if st.button("📝 .streamlit/config.toml Oluştur"):
        streamlit_dir = Path.cwd() / '.streamlit'
        streamlit_dir.mkdir(exist_ok=True)
        
        config_content = """
[server]
maxUploadSize = 500
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 8502

[browser]
gatherUsageStats = false

[global]
developmentMode = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
"""
        
        config_path = streamlit_dir / 'config.toml'
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        
        st.success(f"✅ Konfigürasyon dosyası oluşturuldu: {config_path}")
    
    # Quick Start Commands
    st.header("🚀 Hızlı Başlangıç Komutları")
    
    commands = {
        "Virtual Environment Oluştur": "python3 -m venv streamlit-env",
        "Virtual Environment Aktive Et": "source streamlit-env/bin/activate",
        "Requirements Yükle": "pip install -r requirements.txt",
        "Streamlit Başlat": "streamlit run streamlit_app.py --server.port 8502",
        "Docker ile Başlat": "docker-compose -f docker-compose.streamlit.yml up -d streamlit-dashboard"
    }
    
    for description, command in commands.items():
        with st.expander(f"📋 {description}"):
            st.code(command, language='bash')
            if st.button(f"📋 Kopyala - {description}", key=description):
                st.write("✅ Komut panoya kopyalandı!")
    
    # Port Check
    st.header("🔌 Port Durumu")
    
    if st.button("🔍 Port Kontrolü"):
        import socket
        
        ports_to_check = [8501, 8502, 8503, 8080, 5432, 6379]
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                st.warning(f"⚠️ Port {port}: BUSY")
            else:
                st.success(f"✅ Port {port}: FREE")
    
    # Logs
    st.header("📋 Loglar")
    
    log_dir = Path.cwd() / 'logs'
    if log_dir.exists():
        log_files = list(log_dir.glob('*.log'))
        if log_files:
            selected_log = st.selectbox("Log dosyası seçin:", log_files)
            if st.button("📖 Log Görüntüle"):
                try:
                    with open(selected_log, 'r') as f:
                        log_content = f.read()
                    st.text_area("Log İçeriği", log_content, height=300)
                except Exception as e:
                    st.error(f"Log okunamadı: {e}")
        else:
            st.info("Henüz log dosyası yok")
    else:
        st.info("logs/ dizini bulunamadı")

if __name__ == "__main__":
    main()
