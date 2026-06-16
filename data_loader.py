"""
Data Loader for Drug Side Effect Prediction
Loads and preprocesses SMILES and side effect data
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from config import (
    SMILES_FILE, SIDE_EFFECTS_FILE, DRUG_NAMES_FILE,
    PROCESSED_DIR, DRUG_LOOKUP_PATH
)
import os


class DataLoader:
    def __init__(self):
        """Initialize data loader"""
        self.smiles_df = None
        self.side_effects_df = None
        self.drug_names_df = None
        self.merged_data = None
        self.drug_lookup = {}
    
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        
        # Load SMILES data
        self.smiles_df = pd.read_csv(SMILES_FILE)
        print(f"Loaded {len(self.smiles_df)} drugs with SMILES")
        
        # Load side effects data
        self.side_effects_df = pd.read_csv(SIDE_EFFECTS_FILE)
        # Strip column names to remove leading/trailing spaces
        self.side_effects_df.columns = self.side_effects_df.columns.str.strip()
        print(f"Loaded {len(self.side_effects_df)} side effect records")
        
        # Load drug names
        try:
            self.drug_names_df = pd.read_csv(DRUG_NAMES_FILE)
            # Strip column names
            self.drug_names_df.columns = self.drug_names_df.columns.str.strip()
            print(f"Loaded {len(self.drug_names_df)} drug name mappings")
        except Exception as e:
            print(f"Warning: Could not load drug names file: {e}")
            self.drug_names_df = None
    
    def preprocess_data(self):
        """Preprocess and merge datasets"""
        print("\nPreprocessing data...")
        
        # Get column names from SMILES dataframe
        smiles_cols = self.smiles_df.columns.tolist()
        print(f"SMILES columns: {smiles_cols}")
        
        # Assuming first column is drug name/ID and second is SMILES
        drug_col = smiles_cols[0]
        smiles_col = smiles_cols[1]
        
        # Clean SMILES data
        self.smiles_df = self.smiles_df.dropna(subset=[smiles_col])
        print(f"After removing null SMILES: {len(self.smiles_df)} drugs")
        
        # Get side effects column names
        se_cols = self.side_effects_df.columns.tolist()
        print(f"Side effects columns: {se_cols}")
        
        # The meddra_all_se.csv format: each row is a drug-side effect pair
        # Columns are typically: [drug_id, another_id, code, type, code2, side_effect_name]
        # We need the first column (drug ID) and last column (side effect name)
        
        drug_id_col = se_cols[0].strip()  # First column is drug ID
        side_effect_name_col = se_cols[-1].strip()  # Last column is side effect name
        
        print(f"Using drug ID column: '{drug_id_col}'")
        print(f"Using side effect name column: '{side_effect_name_col}'")
        
        # Group by drug ID and collect all side effects
        print("\nAggregating side effects by drug...")
        drug_side_effects = {}
        
        for _, row in self.side_effects_df.iterrows():
            drug_id = str(row[drug_id_col]).strip()
            side_effect = str(row[side_effect_name_col]).strip()
            
            # Skip if side effect is empty or NaN
            if not side_effect or side_effect == 'nan':
                continue
            
            if drug_id not in drug_side_effects:
                drug_side_effects[drug_id] = set()
            
            drug_side_effects[drug_id].add(side_effect)
        
        # Convert sets to lists
        drug_side_effects = {k: list(v) for k, v in drug_side_effects.items()}
        
        print(f"Found side effects for {len(drug_side_effects)} unique drugs")
        if len(drug_side_effects) > 0:
            sample_drug = list(drug_side_effects.keys())[0]
            print(f"Sample: {sample_drug} has {len(drug_side_effects[sample_drug])} side effects")
        
        # Create a CID to drug name mapping from drug_names.csv
        cid_to_name = {}
        name_to_cid = {}
        if self.drug_names_df is not None:
            try:
                name_cols = self.drug_names_df.columns.tolist()
                if len(name_cols) >= 2:
                    cid_col = name_cols[0].strip()
                    name_col = name_cols[1].strip()
                    
                    for _, row in self.drug_names_df.iterrows():
                        cid = str(row[cid_col]).strip()
                        name = str(row[name_col]).strip().lower()
                        cid_to_name[cid] = name
                        name_to_cid[name] = cid
                    
                    print(f"Created CID mapping for {len(cid_to_name)} drugs")
            except Exception as e:
                print(f"Warning: Could not create CID mapping: {e}")
        
        # Create merged dataset
        # Try to match SMILES drugs with side effect drugs
        merged_records = []
        
        for _, row in self.smiles_df.iterrows():
            drug_name = str(row[drug_col]).strip()
            smiles = str(row[smiles_col]).strip()
            
            # Try different matching strategies
            side_effects = []
            matched_id = None
            
            # Strategy 1: Direct match with drug ID
            if drug_name in drug_side_effects:
                side_effects = drug_side_effects[drug_name]
                matched_id = drug_name
            
            # Strategy 2: Look up drug name in drug_names.csv to get CID
            if not side_effects and drug_name.lower() in name_to_cid:
                cid = name_to_cid[drug_name.lower()]
                if cid in drug_side_effects:
                    side_effects = drug_side_effects[cid]
                    matched_id = cid
            
            # Strategy 3: Case-insensitive match
            if not side_effects:
                drug_name_lower = drug_name.lower()
                for drug_id in drug_side_effects.keys():
                    if drug_id.lower() == drug_name_lower:
                        side_effects = drug_side_effects[drug_id]
                        matched_id = drug_id
                        break
            
            # Strategy 4: Partial match (drug name contains ID or vice versa)
            if not side_effects:
                drug_name_lower = drug_name.lower()
                for drug_id in drug_side_effects.keys():
                    drug_id_lower = drug_id.lower()
                    # Check if CID is in the drug name or vice versa
                    if 'cid' in drug_id_lower and drug_id_lower in drug_name_lower:
                        side_effects = drug_side_effects[drug_id]
                        matched_id = drug_id
                        break
            
            if side_effects and len(side_effects) > 0:  # Only include drugs with known side effects
                merged_records.append({
                    'drug_name': drug_name,
                    'smiles': smiles,
                    'side_effects': side_effects,
                    'matched_id': matched_id
                })
                
                # Add to drug lookup
                self.drug_lookup[drug_name.lower()] = {
                    'name': drug_name,
                    'smiles': smiles
                }
        
        self.merged_data = pd.DataFrame(merged_records)
        print(f"\nMerged dataset: {len(self.merged_data)} drugs with both SMILES and side effects")
        
        # Print statistics
        if len(self.merged_data) > 0:
            avg_side_effects = self.merged_data['side_effects'].apply(len).mean()
            print(f"Average side effects per drug: {avg_side_effects:.2f}")
            
            # Get all unique side effects
            all_side_effects = set()
            for se_list in self.merged_data['side_effects']:
                all_side_effects.update(se_list)
            print(f"Total unique side effects: {len(all_side_effects)}")
            
            # Show sample
            print(f"\nSample merged data:")
            sample = self.merged_data.iloc[0]
            print(f"  Drug: {sample['drug_name']}")
            print(f"  Matched ID: {sample['matched_id']}")
            print(f"  SMILES: {sample['smiles'][:50]}...")
            print(f"  Side effects ({len(sample['side_effects'])}): {sample['side_effects'][:5]}...")
        else:
            print("\n⚠️ WARNING: No drugs matched between SMILES and side effects datasets!")
            print("This might be due to different ID formats. Checking dataset formats...")
            print(f"\nSample SMILES drug names: {self.smiles_df[drug_col].head(5).tolist()}")
            print(f"\nSample side effect drug IDs: {list(drug_side_effects.keys())[:5]}")

    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets (drug-based splitting)
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            train_df, test_df
        """
        # Shuffle data
        shuffled = self.merged_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split
        n_test = int(len(shuffled) * test_size)
        test_df = shuffled[:n_test]
        train_df = shuffled[n_test:]
        
        print(f"\nTrain set: {len(train_df)} drugs")
        print(f"Test set: {len(test_df)} drugs")
        
        return train_df, test_df
    
    def get_kfold_splits(self, n_folds=3, random_state=42):
        """
        Get K-fold cross-validation splits
        
        Args:
            n_folds: Number of folds
            random_state: Random seed
            
        Returns:
            List of (train_df, val_df) tuples
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, val_idx in kf.split(self.merged_data):
            train_df = self.merged_data.iloc[train_idx]
            val_df = self.merged_data.iloc[val_idx]
            splits.append((train_df, val_df))
        
        print(f"\nCreated {n_folds}-fold cross-validation splits")
        return splits
    
    def save_drug_lookup(self, filepath=DRUG_LOOKUP_PATH):
        """Save drug name to SMILES lookup"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.drug_lookup, f)
        print(f"Drug lookup saved to {filepath}")
    
    @staticmethod
    def load_drug_lookup(filepath=DRUG_LOOKUP_PATH):
        """Load drug lookup"""
        with open(filepath, 'rb') as f:
            lookup = pickle.load(f)
        print(f"Drug lookup loaded from {filepath}")
        return lookup


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    loader.load_data()
    loader.preprocess_data()
    
    # Get splits
    train_df, test_df = loader.get_train_test_split()
    
    # Get K-fold splits
    splits = loader.get_kfold_splits(n_folds=3)
    
    # Save drug lookup
    loader.save_drug_lookup()
    
    print("\nData Loader test completed successfully!")
    print(f"\nSample data:")
    print(train_df.head())
