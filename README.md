# Drug Side Effect Prediction using DeepSide Framework

A deep learning system that predicts drug side effects from SMILES molecular representations using CNN architecture. Built with TensorFlow/Keras and Flask.

## 🎯 Project Overview

This project implements the **DeepSide framework** for predicting adverse drug reactions (ADRs) before clinical trials. It uses:
- **Input**: SMILES (Simplified Molecular Input Line Entry System) representations
- **Model**: Convolutional Neural Network (CNN) with multiple Conv1D layers
- **Output**: Multi-label predictions of potential side effects with probabilities

## 📊 Key Features

- ✅ One-hot encoding of SMILES molecular structures
- ✅ Multi-label side effect prediction
- ✅ Weighted binary cross-entropy loss (handles class imbalance)
- ✅ 3-fold cross-validation support
- ✅ Comprehensive evaluation metrics (Micro/Macro AUC, mAP, Hamming Loss)
- ✅ Modern web interface with autocomplete
- ✅ Real-time predictions with precautions

## 🏗️ Project Structure

```
Drug Side effect predection/
│
├── Dataset/                      # Dataset files
│   ├── Name_and_SMILES.csv      # Drug names and SMILES
│   ├── meddra_all_se.csv        # Side effects data
│   └── drug_names.csv           # Drug name mappings
│
├── preprocessing/               # Data preprocessing modules
│   ├── data_loader.py          # Load and merge datasets
│   ├── smiles_encoder.py       # SMILES one-hot encoding
│   └── side_effect_encoder.py  # Multi-label encoding
│
├── model/                       # Model components
│   ├── smiles_cnn.py           # CNN architecture
│   ├── weighted_loss.py        # Custom loss functions
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── templates/                   # HTML templates
│   └── index.html              # Web interface
│
├── static/                      # Static files
│   ├── style.css               # CSS styling
│   └── script.js               # JavaScript
│
├── processed_data/              # Generated during training
│   ├── smiles_encoder.pkl      # Saved encoder
│   ├── side_effect_encoder.pkl # Saved encoder
│   └── drug_lookup.pkl         # Drug name lookup
│
├── venv/                        # Virtual environment
├── config.py                    # Configuration
├── app.py                       # Flask web application
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Installation

### 1. Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📚 Usage

### Step 1: Train the Model

Train the model using simple train/test split (faster):

```bash
python model/train.py --mode simple
```

Or train with 3-fold cross-validation (more robust):

```bash
python model/train.py --mode cv --folds 3
```

**Expected output:**
- Model weights saved to `model/model_weights.h5`
- Model architecture saved to `model/model_architecture.json`
- Encoders saved to `processed_data/`

### Step 2: Evaluate the Model

```bash
python model/evaluate.py
```

**Metrics calculated:**
- ✓ Micro AUC (overall performance)
- ✓ Macro AUC (rare side effects)
- ✓ mAP (mean Average Precision)
- ✓ Hamming Loss
- ✓ Overall accuracy

### Step 3: Run the Web Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## 🎨 Web Interface Features

- **Drug Search**: Autocomplete drug name input
- **Predictions**: Top 5 side effects with probability scores
- **Severity Indicators**: Color-coded (High/Medium/Low)
- **Precautions**: Personalized recommendations based on predictions
- **Modern UI**: Dark theme with smooth animations

## 🧠 Model Architecture

### SMILESConv CNN

```
Input: One-hot encoded SMILES (200 x vocab_size)
    ↓
Conv1D (64 filters) → ReLU → BatchNorm → MaxPool → Dropout
    ↓
Conv1D (128 filters) → ReLU → BatchNorm → MaxPool → Dropout
    ↓
Conv1D (256 filters) → ReLU → BatchNorm → MaxPool → Dropout
    ↓
Flatten
    ↓
Dense (512) → ReLU → BatchNorm → Dropout
    ↓
Dense (256) → ReLU → BatchNorm → Dropout
    ↓
Output: Dense (n_side_effects) → Sigmoid
```

### Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Weighted Binary Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation**: 3-fold cross-validation

## 📈 Evaluation Metrics

### Required for Viva/Report

1. **Micro AUC**: Overall performance across all side effects
2. **Macro AUC**: Performance on rare side effects
3. **mAP**: Mean Average Precision
4. **Hamming Loss**: Multi-label classification error

### Interpretation

- **Micro AUC > 0.7**: Acceptable performance
- **Macro AUC > 0.6**: Good handling of rare side effects
- **Hamming Loss < 0.1**: Low error rate

## 🔧 Configuration

Edit `config.py` to customize:

- Model hyperparameters (filters, layers, dropout)
- Training parameters (batch size, epochs, learning rate)
- SMILES encoding (max length, character set)
- Web app settings (host, port)

## 📝 Dataset Information

### Input Files

1. **Name_and_SMILES.csv**: ~1,400 drugs with molecular structures
2. **meddra_all_se.csv**: Drug-side effect relationships
3. **drug_names.csv**: Drug identifier mappings

### Data Processing

- SMILES sequences are one-hot encoded (max length: 200 characters)
- Side effects are multi-label encoded
- Class weights calculated to handle imbalanced data
- Drug-based splitting (no data leakage)

## 🎯 Example Predictions

```
Drug: Aspirin
Predicted Side Effects:
  1. Gastrointestinal bleeding (85.2%)
  2. Nausea (72.4%)
  3. Headache (68.1%)
  4. Dizziness (54.3%)
  ...

Precautions:
  ⚠️ Take medication with food to reduce nausea
  ⚠️ Monitor for signs of bleeding
  ⚠️ Consult your healthcare provider
```

## 🛠️ Troubleshooting

### Issue: Model not found
**Solution**: Run training script first (`python model/train.py`)

### Issue: Drug not found
**Solution**: Check drug name spelling or try autocomplete suggestions

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and dependencies installed

## 📚 References

- DeepSide Framework: Deep Learning for Drug Side Effect Prediction
- SMILES Notation: Molecular structure representation
- MedDRA: Medical Dictionary for Regulatory Activities

## 👥 Contributors

Built for Batch 8 Deep Learning Project

## 📄 License

This project is for educational purposes.

## ⚠️ Disclaimer

This is a prediction tool for educational and research purposes only. Always consult qualified healthcare professionals before taking any medication. The predictions are based on historical data and should not replace professional medical advice.
