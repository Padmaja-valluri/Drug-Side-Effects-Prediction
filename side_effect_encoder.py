"""
Side Effect Encoder for Drug Side Effect Prediction
Handles multi-label encoding of side effects
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from config import SIDE_EFFECT_ENCODER_PATH


class SideEffectEncoder:
    def __init__(self):
        """Initialize side effect encoder"""
        self.mlb = MultiLabelBinarizer()
        self.side_effects = []
        self.n_side_effects = 0
        self.class_weights = None
    
    def fit(self, side_effects_list):
        """
        Fit encoder on side effects data
        
        Args:
            side_effects_list: List of lists, where each inner list contains side effects for a drug
        """
        self.mlb.fit(side_effects_list)
        self.side_effects = list(self.mlb.classes_)
        self.n_side_effects = len(self.side_effects)
        
        # Calculate class weights for weighted loss
        self._calculate_class_weights(side_effects_list)
        
        print(f"Fitted encoder with {self.n_side_effects} unique side effects")
    
    def _calculate_class_weights(self, side_effects_list):
        """
        Calculate class weights based on frequency
        Rare side effects get higher weights
        """
        # Transform to binary matrix
        binary_matrix = self.mlb.transform(side_effects_list)
        
        # Calculate frequency of each side effect
        frequencies = binary_matrix.sum(axis=0)
        total_samples = len(side_effects_list)
        
        # Calculate weights (inverse frequency)
        # Add small epsilon to avoid division by zero
        weights = total_samples / (frequencies + 1e-6)
        
        # Normalize weights
        self.class_weights = weights / weights.sum() * len(weights)
        
        print(f"Calculated class weights - Min: {self.class_weights.min():.2f}, Max: {self.class_weights.max():.2f}")
    
    def encode(self, side_effects):
        """
        Encode side effects to binary vector
        
        Args:
            side_effects: List of side effects for a single drug
            
        Returns:
            Binary vector of shape (n_side_effects,)
        """
        return self.mlb.transform([side_effects])[0]
    
    def encode_batch(self, side_effects_list):
        """
        Encode batch of side effects
        
        Args:
            side_effects_list: List of lists of side effects
            
        Returns:
            Binary matrix of shape (batch_size, n_side_effects)
        """
        return self.mlb.transform(side_effects_list)
    
    def decode(self, binary_vector, threshold=0.5):
        """
        Decode binary vector back to side effects
        
        Args:
            binary_vector: Binary or probability vector
            threshold: Threshold for considering a side effect present
            
        Returns:
            List of side effects
        """
        # Apply threshold
        binary = (binary_vector >= threshold).astype(int)
        
        # Get side effects
        indices = np.where(binary == 1)[0]
        return [self.side_effects[i] for i in indices]
    
    def decode_with_probabilities(self, prob_vector, top_k=None):
        """
        Decode probability vector to side effects with probabilities
        
        Args:
            prob_vector: Probability vector
            top_k: Return only top k predictions (None for all)
            
        Returns:
            List of tuples (side_effect, probability)
        """
        # Get all side effects with probabilities
        results = [(self.side_effects[i], prob_vector[i]) for i in range(len(prob_vector))]
        
        # Sort by probability (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k if specified
        if top_k:
            results = results[:top_k]
        
        return results
    
    def get_class_weights_dict(self):
        """
        Get class weights as dictionary for Keras
        
        Returns:
            Dictionary mapping class index to weight
        """
        if self.class_weights is None:
            return None
        return {i: self.class_weights[i] for i in range(len(self.class_weights))}
    
    def save(self, filepath=SIDE_EFFECT_ENCODER_PATH):
        """Save encoder to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Side effect encoder saved to {filepath}")
    
    @staticmethod
    def load(filepath=SIDE_EFFECT_ENCODER_PATH):
        """Load encoder from file"""
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
        print(f"Side effect encoder loaded from {filepath}")
        return encoder


if __name__ == "__main__":
    # Test the encoder
    test_data = [
        ['Headache', 'Nausea', 'Dizziness'],
        ['Headache', 'Fatigue'],
        ['Nausea', 'Vomiting', 'Diarrhea'],
        ['Headache', 'Dizziness']
    ]
    
    encoder = SideEffectEncoder()
    encoder.fit(test_data)
    
    # Test encoding
    test_se = ['Headache', 'Nausea']
    encoded = encoder.encode(test_se)
    print(f"\nEncoded {test_se}: {encoded}")
    
    # Test decoding
    decoded = encoder.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Test with probabilities
    prob_vector = np.array([0.8, 0.3, 0.6, 0.1, 0.9])
    results = encoder.decode_with_probabilities(prob_vector, top_k=3)
    print(f"\nTop 3 predictions:")
    for se, prob in results:
        print(f"  {se}: {prob:.2f}")
    
    print("\nSide Effect Encoder test completed successfully!")
