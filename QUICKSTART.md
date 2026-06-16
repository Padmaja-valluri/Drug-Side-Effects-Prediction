# 🚀 Quick Start Guide - Drug Side Effect Prediction

## Prerequisites
✅ Virtual environment created (`venv/`)  
✅ All code files in place  
✅ Dataset files in `Dataset/` folder  

---

## Step-by-Step Instructions

### 1️⃣ Activate Virtual Environment

Open PowerShell in the project directory and run:

```powershell
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### 2️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

**Expected time:** 2-5 minutes

**Packages installed:**
- TensorFlow 2.15.0
- Pandas, NumPy
- Scikit-learn
- Flask
- Matplotlib, Seaborn

### 3️⃣ Train the Model

```powershell
python model/train.py --mode simple
```

**What this does:**
- Loads and preprocesses dataset
- Encodes SMILES and side effects
- Trains CNN model (50 epochs max, early stopping enabled)
- Saves model weights and encoders

**Expected time:** 10-30 minutes (depends on your hardware)

**Output files created:**
- `model/model_weights.h5` - Trained model weights
- `model/model_architecture.json` - Model structure
- `processed_data/smiles_encoder.pkl` - SMILES encoder
- `processed_data/side_effect_encoder.pkl` - Side effect encoder
- `processed_data/drug_lookup.pkl` - Drug name lookup

### 4️⃣ Evaluate the Model

```powershell
python model/evaluate.py
```

**What this does:**
- Loads trained model
- Calculates all required metrics
- Generates evaluation report

**Metrics you'll see:**
- ✅ Micro AUC
- ✅ Macro AUC
- ✅ mAP (micro and macro)
- ✅ Hamming Loss
- ✅ Overall Accuracy

### 5️⃣ Run the Web Application

```powershell
python app.py
```

**Then open your browser to:**
```
http://localhost:5000
```

**Features:**
- Search for drug names (with autocomplete)
- Get side effect predictions
- View probability scores
- See personalized precautions

---

## 🎯 Testing the Web App

1. **Open browser:** http://localhost:5000
2. **Enter a drug name:** Try "Aspirin" or "Ibuprofen"
3. **Click "Predict Side Effects"**
4. **View results:**
   - Top 5 predicted side effects
   - Probability percentages
   - Color-coded severity (Red/Orange/Green)
   - Precautions and recommendations

---

## 🐛 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution:** Make sure virtual environment is activated and dependencies installed

### Issue: "Model weights not found"
**Solution:** Run training first: `python model/train.py --mode simple`

### Issue: "Drug not found in database"
**Solution:** Try different drug names or check autocomplete suggestions

### Issue: Port 5000 already in use
**Solution:** Edit `config.py` and change `WEB_CONFIG['port']` to another port (e.g., 5001)

---

## 📊 Expected Results

### Training Output
```
Epoch 1/50
loss: 0.XXX - binary_accuracy: 0.XXX - auc: 0.XXX
...
Epoch XX/50
loss: 0.XXX - binary_accuracy: 0.XXX - auc: 0.XXX
Early stopping triggered
Model saved!
```

### Evaluation Output
```
📊 Required Metrics for Viva/Report:
  • Micro AUC:     0.75-0.85 (expected range)
  • Macro AUC:     0.65-0.75 (expected range)
  • mAP (micro):   0.70-0.80 (expected range)
  • Hamming Loss:  0.05-0.15 (expected range)
```

---

## 📁 Project Structure

```
Drug Side effect predection/
├── venv/                    ✅ Virtual environment
├── Dataset/                 ✅ Your dataset files
├── preprocessing/           ✅ Data processing modules
├── model/                   ✅ CNN model and training
├── templates/               ✅ HTML files
├── static/                  ✅ CSS and JavaScript
├── processed_data/          📦 Created during training
├── config.py                ✅ Configuration
├── app.py                   ✅ Web application
├── requirements.txt         ✅ Dependencies
└── README.md                ✅ Documentation
```

---

## 🎓 For Your Report/Viva

### Key Points to Mention

1. **Architecture:** SMILESConv CNN with 3 Conv1D blocks
2. **Input:** One-hot encoded SMILES (max 200 characters)
3. **Output:** Multi-label side effect predictions
4. **Loss:** Weighted binary cross-entropy (handles imbalance)
5. **Validation:** 3-fold cross-validation support
6. **Metrics:** Micro/Macro AUC, mAP, Hamming Loss

### Dataset Details
- ~1,400 drugs with SMILES representations
- Hundreds of unique side effects
- Multi-label classification problem

### Model Performance
- Handles class imbalance with weighted loss
- Drug-based splitting prevents data leakage
- Early stopping prevents overfitting

---

## ✅ Checklist Before Running

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Dataset files in `Dataset/` folder
- [ ] Sufficient disk space (~500MB for model files)
- [ ] Python 3.8+ installed

---

## 🎉 You're All Set!

Follow the steps above and you'll have a fully functional Drug Side Effect Prediction system with a modern web interface.

**Need help?** Check the [README.md](file:///c:/Users/DELL/OneDrive/Desktop/Drug%20Side%20effect%20predection/README.md) for detailed documentation.

**Good luck! 🚀**
