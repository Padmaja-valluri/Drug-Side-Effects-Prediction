"""
SMILES Encoder for Drug Side Effect Prediction
Handles one-hot encoding of SMILES molecular representations
"""
import numpy as np
import pickle
from config import MAX_SMILES_LENGTH, SMILES_CHARSET, SMILES_ENCODER_PATH


class SMILESEncoder:
    def __init__(self, max_length=MAX_SMILES_LENGTH, charset=None):
        """
        Initialize SMILES encoder
        
        Args:
            max_length: Maximum length of SMILES sequences
            charset: List of characters in SMILES vocabulary
        """
        self.max_length = max_length
        self.charset = charset if charset else SMILES_CHARSET
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        self.vocab_size = len(self.charset)
    
    def encode(self, smiles):
        """
        One-hot encode a SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            One-hot encoded array of shape (max_length, vocab_size)
        """
        # Initialize zero matrix
        encoded = np.zeros((self.max_length, self.vocab_size), dtype=np.float32)
        
        # Truncate or pad SMILES
        smiles = smiles[:self.max_length]
        
        # One-hot encode each character
        for i, char in enumerate(smiles):
            if char in self.char_to_idx:
                encoded[i, self.char_to_idx[char]] = 1.0
        
        return encoded
    
    def encode_batch(self, smiles_list):
        """
        Encode a batch of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Numpy array of shape (batch_size, max_length, vocab_size)
        """
        return np.array([self.encode(smiles) for smiles in smiles_list])
    
    def decode(self, encoded):
        """
        Decode a one-hot encoded SMILES back to string
        
        Args:
            encoded: One-hot encoded array of shape (max_length, vocab_size)
            
        Returns:
            SMILES string
        """
        chars = []
        for i in range(encoded.shape[0]):
            idx = np.argmax(encoded[i])
            if encoded[i, idx] > 0:  # Only add if not padding
                chars.append(self.idx_to_char.get(idx, ''))
        
        return ''.join(chars)
    
    def save(self, filepath=SMILES_ENCODER_PATH):
        """Save encoder to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"SMILES encoder saved to {filepath}")
    
    @staticmethod
    def load(filepath=SMILES_ENCODER_PATH):
        """Load encoder from file"""
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
        print(f"SMILES encoder loaded from {filepath}")
        return encoder


if __name__ == "__main__":
    # Test the encoder
    encoder = SMILESEncoder()
    
    # Test SMILES
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    print(f"Original SMILES: {test_smiles}")
    
    # Encode
    encoded = encoder.encode(test_smiles)
    print(f"Encoded shape: {encoded.shape}")
    
    # Decode
    decoded = encoder.decode(encoded)
    print(f"Decoded SMILES: {decoded}")
    
    # Save encoder
    encoder.save()
    print("\nSMILES Encoder test completed successfully!")
