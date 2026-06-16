"""
Flask Web Application for Drug Side Effect Prediction
"""
import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.smiles_encoder import SMILESEncoder
from preprocessing.side_effect_encoder import SideEffectEncoder
from preprocessing.data_loader import DataLoader
from model.smiles_cnn import build_smiles_cnn
from config import (
    MODEL_WEIGHTS_PATH, SMILES_ENCODER_PATH,
    SIDE_EFFECT_ENCODER_PATH, DRUG_LOOKUP_PATH,
    WEB_CONFIG, MAX_SMILES_LENGTH, EVAL_CONFIG
)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and encoders
model = None
smiles_encoder = None
side_effect_encoder = None
drug_lookup = None


def load_model_and_encoders():
    """Load trained model and encoders"""
    global model, smiles_encoder, side_effect_encoder, drug_lookup
    
    print("Loading encoders...")
    smiles_encoder = SMILESEncoder.load(SMILES_ENCODER_PATH)
    side_effect_encoder = SideEffectEncoder.load(SIDE_EFFECT_ENCODER_PATH)
    
    print("Loading drug lookup...")
    drug_lookup = DataLoader.load_drug_lookup(DRUG_LOOKUP_PATH)
    
    print("Loading model...")
    input_shape = (MAX_SMILES_LENGTH, smiles_encoder.vocab_size)
    n_outputs = side_effect_encoder.n_side_effects
    
    model = build_smiles_cnn(input_shape, n_outputs)
    model.load_weights(MODEL_WEIGHTS_PATH)
    
    print("Model and encoders loaded successfully!")


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    """Get list of available drugs for autocomplete"""
    try:
        # Get search query
        query = request.args.get('q', '').lower()
        
        # Filter drugs
        if query:
            matching_drugs = [
                {'name': info['name'], 'smiles': info['smiles'][:50] + '...'}
                for drug_name, info in drug_lookup.items()
                if query in drug_name
            ]
        else:
            # Return first 50 drugs
            matching_drugs = [
                {'name': info['name'], 'smiles': info['smiles'][:50] + '...'}
                for drug_name, info in list(drug_lookup.items())[:50]
            ]
        
        return jsonify({
            'success': True,
            'drugs': matching_drugs[:20]  # Limit to 20 results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict side effects for a drug"""
    try:
        # Get drug name from request
        data = request.get_json()
        drug_name = data.get('drug_name', '').strip()
        
        if not drug_name:
            return jsonify({
                'success': False,
                'error': 'Please provide a drug name'
            })
        
        # Look up SMILES
        drug_name_lower = drug_name.lower()
        
        if drug_name_lower not in drug_lookup:
            return jsonify({
                'success': False,
                'error': f'Drug "{drug_name}" not found in database. Please try another drug name.'
            })
        
        smiles = drug_lookup[drug_name_lower]['smiles']
        actual_name = drug_lookup[drug_name_lower]['name']
        
        # Encode SMILES
        X = smiles_encoder.encode(smiles)
        X = np.expand_dims(X, axis=0)  # Add batch dimension
        
        # Predict
        y_pred = model.predict(X, verbose=0)[0]
        
        # Decode predictions
        top_k = EVAL_CONFIG['top_k_predictions']
        predictions = side_effect_encoder.decode_with_probabilities(y_pred, top_k=top_k)
        
        # Format results
        side_effects = []
        for side_effect, probability in predictions:
            side_effects.append({
                'name': side_effect,
                'probability': float(probability),
                'percentage': float(probability * 100),
                'severity': get_severity(probability)
            })
        
        # Generate precautions
        precautions = generate_precautions(side_effects)
        
        return jsonify({
            'success': True,
            'drug_name': actual_name,
            'smiles': smiles,
            'side_effects': side_effects,
            'precautions': precautions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        })


def get_severity(probability):
    """Get severity level based on probability"""
    if probability >= 0.7:
        return 'high'
    elif probability >= 0.4:
        return 'medium'
    else:
        return 'low'


def generate_precautions(side_effects):
    """Generate precautions based on predicted side effects"""
    precautions = []
    
    # General precautions
    precautions.append("Consult your healthcare provider before taking this medication")
    precautions.append("Follow the prescribed dosage and schedule")
    
    # Specific precautions based on side effects
    if any('nausea' in se['name'].lower() or 'vomiting' in se['name'].lower() 
           for se in side_effects[:3]):
        precautions.append("Take medication with food to reduce nausea")
    
    if any('dizziness' in se['name'].lower() or 'drowsiness' in se['name'].lower() 
           for se in side_effects[:3]):
        precautions.append("Avoid driving or operating heavy machinery")
    
    if any('headache' in se['name'].lower() for se in side_effects[:3]):
        precautions.append("Stay hydrated and get adequate rest")
    
    if any('rash' in se['name'].lower() or 'itching' in se['name'].lower() 
           for se in side_effects[:3]):
        precautions.append("Monitor for allergic reactions and seek immediate medical attention if severe")
    
    # High probability side effects
    high_prob_effects = [se for se in side_effects if se['severity'] == 'high']
    if high_prob_effects:
        precautions.append("Be aware of high-probability side effects and report any concerns to your doctor")
    
    precautions.append("Keep a record of any side effects you experience")
    precautions.append("Do not stop taking medication without consulting your doctor")
    
    return precautions


if __name__ == '__main__':
    # Load model and encoders
    load_model_and_encoders()
    
    # Run app
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug']
    )
