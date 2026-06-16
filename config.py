"""
Configuration file for Drug Side Effect Prediction
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'Dataset')
SMILES_FILE = os.path.join(DATA_DIR, 'Name_and_SMILES.csv')
SIDE_EFFECTS_FILE = os.path.join(DATA_DIR, 'meddra_all_se.csv')
DRUG_NAMES_FILE = os.path.join(DATA_DIR, 'drug_names.csv')

# Processed data paths
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'model_weights.weights.h5')
MODEL_ARCHITECTURE_PATH = os.path.join(MODEL_DIR, 'model_architecture.json')

# Encoder paths
SMILES_ENCODER_PATH = os.path.join(PROCESSED_DIR, 'smiles_encoder.pkl')
SIDE_EFFECT_ENCODER_PATH = os.path.join(PROCESSED_DIR, 'side_effect_encoder.pkl')
DRUG_LOOKUP_PATH = os.path.join(PROCESSED_DIR, 'drug_lookup.pkl')

# SMILES encoding parameters
MAX_SMILES_LENGTH = 200
SMILES_CHARSET = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si',
    'c', 'n', 'o', 's', 'p',
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '(', ')', '[', ']', '=', '#', '@', '+', '-', '/', '\\', '%'
]

# Model hyperparameters
MODEL_CONFIG = {
    'conv_filters': [64, 128, 256],
    'kernel_sizes': [3, 5, 7],
    'pool_size': 2,
    'dense_units': [512, 256],
    'dropout_rate': 0.5,
    'activation': 'relu',
    'output_activation': 'sigmoid'
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'n_folds': 3,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5
}

# Evaluation parameters
EVAL_CONFIG = {
    'threshold': 0.5,
    'top_k_predictions': 5
}

# Web app configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}
