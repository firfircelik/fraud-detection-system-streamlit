# 🤝 Contributing to Fraud Detection System

Thank you for your interest in contributing! 🎉

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git`
3. **Create** a branch: `git checkout -b feature/amazing-feature`
4. **Install** dependencies: `pip install -r backend/requirements.txt`
5. **Make** your changes
6. **Test** locally: `uvicorn api.main:app --reload` (backend) and `streamlit run streamlit_app.py` (frontend)
7. **Commit** changes: `git commit -m 'Add amazing feature'`
8. **Push** to branch: `git push origin feature/amazing-feature`
9. **Create** Pull Request

## 💡 How to Contribute

### 🐛 Bug Reports
- Use GitHub Issues
- Include steps to reproduce
- Add screenshots if helpful

### ✨ Feature Requests  
- Open an Issue first to discuss
- Explain the use case
- Consider implementation approach

### 🔧 Code Contributions
- Follow existing code style
- Add comments for complex logic
- Update README if needed
- Test your changes

## 📋 Development Setup

```bash
# 1. Clone and setup
git clone https://github.com/firfircelik/fraud-detection-system.git
cd fraud-detection-system

# 2. Create virtual environment
python -m venv fraud-env
source fraud-env/bin/activate  # On Windows: fraud-env\Scripts\activate

# 3. Install backend dependencies
cd backend
pip install -r requirements.txt

# 4. Run backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Run Streamlit frontend (in new terminal)
streamlit run streamlit_app.py --server.port 8501
```

## 🎯 Areas for Contribution

- 🔍 **New fraud detection algorithms**
- 📊 **Additional visualization types**
- 🌐 **Internationalization (i18n)**
- ⚡ **Performance optimizations**
- 📱 **Mobile UI improvements**
- 🧪 **Unit tests**
- 📖 **Documentation**

## 📝 Code Style

- Use meaningful variable names
- Add docstrings for functions
- Keep functions focused and small
- Comment complex algorithms

## ⚡ Testing

Before submitting:
```bash
# Test the backend API
cd backend
uvicorn api.main:app --reload

# Check for Python errors
python -m py_compile api/*.py

# Test the Streamlit frontend
streamlit run streamlit_app.py --server.port 8501

# Test API endpoints
# Visit http://localhost:8000/docs for API documentation
# Test frontend at http://localhost:8501
```

## 🏆 Recognition

Contributors will be:
- ✅ Added to README credits
- 🎉 Mentioned in release notes
- ⭐ Given contributor badge

## 📧 Questions?

- 💬 Open a GitHub Issue
- 📧 Contact [@firfircelik](https://github.com/firfircelik)

---
**Thank you for making this project better! 🚀**
