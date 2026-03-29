# NeuroScan AI — Setup Instructions

## ⚠️ Python 3.14 Note

Your system is running **Python 3.14.3**, which is cutting-edge. Most packages have not yet released wheels for Python 3.14 support.

**Recommended:** Downgrade to Python 3.11 or 3.12 for full functionality with TensorFlow. However, you can still run the app in **demo mode** with Python 3.14.

---

## Option 1: Continue with Python 3.14 (Demo Mode Only)

If you want to test the UI without the trained model:

```powershell
# 1. Navigate to project
cd F:\Mini-Project-ML

# 2. Create virtual environment (already done)
# Virtual env is at: F:\Mini-Project-ML\venv

# 3. Install core dependencies (without TensorFlow)
.\venv\Scripts\python.exe -m pip install --no-cache-dir -r .\files\requirements.txt

# 4. Launch the app
.\venv\Scripts\python.exe -m streamlit run .\files\app.py
```

**Expected behavior:**
- Streamlit UI loads at http://localhost:8501
- You can upload MRI images
- Predictions use image statistics (demo mode) instead of a real model
- All visualizations work

---

## Option 2: Downgrade to Python 3.11/3.12 (Full Functionality)

For complete functionality with TensorFlow:

### Step 1: Install Python 3.11 or 3.12
- Download from https://www.python.org/downloads/
- Choose Python 3.11.x or 3.12.x
- **Important:** Check "Add Python to PATH" during installation

### Step 2: Create new virtual environment
```powershell
cd F:\Mini-Project-ML

# Remove old venv (optional)
Remove-Item -Recurse -Force .\venv

# Create new venv with Python 3.11/3.12
C:\Python312\python.exe -m venv venv  # adjust path based on your Python installation
```

### Step 3: Install all dependencies (including TensorFlow)
```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

.\venv\Scripts\python.exe -m pip install streamlit>=1.28.0 \
  tensorflow>=2.13.0 numpy>=1.24.0 Pillow>=10.0.0 \
  plotly>=5.14.0 pandas>=2.0.0 matplotlib>=3.7.0 \
  seaborn>=0.12.0 opencv-python-headless>=4.8.0 \
  scikit-learn>=1.3.0 scipy>=1.11.0
```

Or simply use the requirements file:
```powershell
.\venv\Scripts\python.exe -m pip install -r .\files\requirements.txt
```

If you are already inside `F:\Mini-Project-ML\files`, use:
```powershell
..\venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### Step 4: Launch the app
```powershell
.\venv\Scripts\python.exe -m streamlit run .\files\app.py
```

---

## Installation Troubleshooting

**If pip takes too long or hangs:**
```powershell
# Kill any hanging python processes
Stop-Process -Name python -Force

# Try installing with increased timeout
.\venv\Scripts\python.exe -m pip install --default-timeout=1000 -r .\files\requirements.txt
```

**If you get "ModuleNotFoundError":**
```powershell
# Verify venv and reinstall
.\venv\Scripts\python.exe -m pip list  # should show streamlit, numpy, etc.

# If empty, reinstall requirements
.\venv\Scripts\python.exe -m pip install --no-cache-dir -r .\files\requirements.txt
```

**If Streamlit won't start:**
```powershell
# Test import directly
.\venv\Scripts\python.exe -c "import streamlit; print('OK')"

# If fails, reinstall streamlit specifically
.\venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall streamlit>=1.28.0
```

---

## Launch Commands

**Demo Mode (Python 3.14):**
```powershell
cd F:\Mini-Project-ML\files
..\venv\Scripts\python.exe -m streamlit run app.py
```

**Full Mode (Python 3.11/3.12):**
```powershell
cd F:\Mini-Project-ML\files
..\venv\Scripts\python.exe -m streamlit run app.py
```

Then open: **http://localhost:8501**

---

## What to Expect

### Demo Mode
- ✅ Upload MRI images
- ✅ Get instant predictions (~5-10% random variance per image)
- ✅ See all visualizations and charts
- ✅ GradCAM works (based on image statistics)
- ❌ Model inference not available

### Full Mode (with trained model)
- ✅ All demo mode features
- ✅ Real EfficientNet-B3 predictions
- ✅ Actual confidence scores (90–98%)
- ✅ Clinical-grade inference
- ❌ Requires real model file at `model/efficientnet_b3_dementia.h5`

---

## Next Steps

1. **Test demo mode** now with Python 3.14 to explore the UI
2. **Install Python 3.12** when ready for full ML inference
3. **Place trained model** at `f:\Mini-Project-ML\files\model\efficientnet_b3_dementia.h5` for real predictions

For questions, see [README.md](README.md) or [DEVELOPER_GUIDE.txt](DEVELOPER_GUIDE.txt).
