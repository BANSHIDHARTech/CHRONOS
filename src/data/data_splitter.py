"""
Data Splitter Module

Implements time-based train/validation/test splits with strict temporal ordering.
NO random shuffling - ensures no future data leakage.
"""

import logging
from typing import Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Time-based data splitting with validation.
    
    Ensures strict temporal ordering: train < validation < test
    No random shuffling to prevent future data leakage.
    """
    
    def __init__(self, config):
        """
        Initialize DataSplitter with configuration.
        
        Args:
            config: Configuration module or object
        """
        self.config = config
    
    def split_by_date(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data by date ranges.
        
        Args:
            X: Features DataFrame
            y: Target Series
            train_start: Training start date
            train_end: Training end date
            val_start: Validation start date
            val_end: Validation end date
            test_start: Test start date
            test_end: Test end date
            
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing (X, y) tuples
        """
        logger.info("Splitting data by date ranges...")
        
        # Training split
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # Validation split
        val_mask = (X.index >= val_start) & (X.index <= val_end)
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        # Test split
        test_mask = (X.index >= test_start) & (X.index <= test_end)
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        # Validate splits
        self.validate_temporal_split(splits)
        
        # Log split information
        logger.info("\nSplit Summary:")
        logger.info(f"Train: {len(X_train)} samples ({train_start} to {train_end})")
        logger.info(f"  Date range: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Val:   {len(X_val)} samples ({val_start} to {val_end})")
        logger.info(f"  Date range: {X_val.index[0]} to {X_val.index[-1]}")
        logger.info(f"Test:  {len(X_test)} samples ({test_start} to {test_end})")
        logger.info(f"  Date range: {X_test.index[0]} to {X_test.index[-1]}")
        
        # Get split statistics
        stats = self.get_split_statistics(splits)
        
        return splits
    
    def validate_temporal_split(self, splits: Dict) -> bool:
        """
        Validate that splits have proper temporal ordering.
        
        Args:
            splits: Dictionary of splits
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If temporal ordering is violated
        """
        logger.info("Validating temporal split ordering...")
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Check that all splits have data
        if len(X_train) == 0:
            raise ValueError("Training set is empty")
        if len(X_val) == 0:
            raise ValueError("Validation set is empty")
        if len(X_test) == 0:
            raise ValueError("Test set is empty")
        
        # Get date boundaries
        train_max = X_train.index.max()
        val_min = X_val.index.min()
        val_max = X_val.index.max()
        test_min = X_test.index.min()
        
        # Validate temporal ordering
        if train_max >= val_min:
            raise ValueError(
                f"Train/Val overlap detected: "
                f"train_max={train_max}, val_min={val_min}"
            )
        
        if val_max >= test_min:
            raise ValueError(
                f"Val/Test overlap detected: "
                f"val_max={val_max}, test_min={test_min}"
            )
        
        logger.info("✓ Temporal ordering validated: train < val < test")
        
        # Check for gaps between splits
        train_to_val_gap = (val_min - train_max).days
        val_to_test_gap = (test_min - val_max).days
        
        if train_to_val_gap > 7:
            logger.warning(
                f"Large gap between train and val: {train_to_val_gap} days"
            )
        
        if val_to_test_gap > 7:
            logger.warning(
                f"Large gap between val and test: {val_to_test_gap} days"
            )
        
        # Verify X and y alignment
        for split_name, (X, y) in splits.items():
            if not X.index.equals(y.index):
                raise ValueError(f"X and y indices don't match in {split_name} split")
        
        logger.info("✓ X and y alignment validated for all splits")
        
        return True
    
    def get_split_statistics(self, splits: Dict) -> Dict:
        """
        Calculate statistics for each split.
        
        Args:
            splits: Dictionary of splits
            
        Returns:
            Dictionary with statistics for each split
        """
        logger.info("\nCalculating split statistics...")
        
        statistics = {}
        
        for split_name, (X, y) in splits.items():
            stats = {
                'num_samples': len(X),
                'num_features': X.shape[1],
                'date_range': (X.index[0], X.index[-1]),
                'class_distribution': y.value_counts().sort_index().to_dict(),
                'class_percentages': (y.value_counts(normalize=True) * 100).sort_index().to_dict()
            }
            
            statistics[split_name] = stats
            
            # Log class distribution
            logger.info(f"\n{split_name.upper()} Split Class Distribution:")
            for quintile in sorted(stats['class_distribution'].keys()):
                count = stats['class_distribution'][quintile]
                pct = stats['class_percentages'][quintile]
                logger.info(f"  Quintile {quintile}: {count:4d} samples ({pct:5.1f}%)")
            
            # Check for severe class imbalance
            min_pct = min(stats['class_percentages'].values())
            max_pct = max(stats['class_percentages'].values())
            
            if max_pct - min_pct > 10:
                logger.warning(
                    f"{split_name} split has class imbalance: "
                    f"min={min_pct:.1f}%, max={max_pct:.1f}%"
                )
        
        return statistics
    
    def split_with_config(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data using dates from configuration.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with keys 'train', 'val', 'test'
        """
        return self.split_by_date(
            X, y,
            train_start=self.config.TRAIN_START,
            train_end=self.config.TRAIN_END,
            val_start=self.config.VAL_START,
            val_end=self.config.VAL_END,
            test_start=self.config.TEST_START,
            test_end=self.config.TEST_END
        )
    
    def save_splits(
        self,
        splits: Dict,
        base_dir: str
    ):
        """
        Save splits to pickle files.
        
        Args:
            splits: Dictionary of splits
            base_dir: Base directory for saving
        """
        import os
        
        os.makedirs(base_dir, exist_ok=True)
        
        for split_name, (X, y) in splits.items():
            # Save as single DataFrame for convenience
            split_df = pd.concat([X, y.rename('target')], axis=1)
            
            filepath = os.path.join(base_dir, f'{split_name}_split.pkl')
            split_df.to_pickle(filepath)
            logger.info(f"Saved {split_name} split to {filepath}")
    
    def load_splits(
        self,
        base_dir: str
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Load splits from pickle files.
        
        Args:
            base_dir: Base directory containing split files
            
        Returns:
            Dictionary with keys 'train', 'val', 'test'
        """
        import os
        
        splits = {}
        
        for split_name in ['train', 'val', 'test']:
            filepath = os.path.join(base_dir, f'{split_name}_split.pkl')
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Split file not found: {filepath}")
            
            split_df = pd.read_pickle(filepath)
            
            # Separate features and target
            X = split_df.drop('target', axis=1)
            y = split_df['target']
            
            splits[split_name] = (X, y)
            logger.info(f"Loaded {split_name} split from {filepath}")
        
        return splits


# Example usage
if __name__ == '__main__':
    from config import (
        TICKERS, DATA_START_DATE, DATA_END_DATE,
        ALIGNED_DATA_FILE, PROCESSED_DATA_DIR
    )
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from target_engineering import TargetEngineer
    
    # Load or create aligned data
    import os
    
    if os.path.exists(ALIGNED_DATA_FILE):
        logger.info(f"Loading aligned data from {ALIGNED_DATA_FILE}")
        aligned_data = pd.read_pickle(ALIGNED_DATA_FILE)
        
        # Split into X and y
        X = aligned_data.drop('quintile_label', axis=1)
        y = aligned_data['quintile_label']
    else:
        logger.info("Creating aligned data from scratch...")
        
        # Load data
        loader = DataLoader()
        data = loader.get_data(list(TICKERS.keys()), DATA_START_DATE, DATA_END_DATE)
        
        # Engineer features
        import config
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(data)
        
        # Engineer targets
        target_engineer = TargetEngineer(config)
        close_prices = data['SPY']['Close']
        targets = target_engineer.engineer_targets(close_prices)
        
        # Align
        X, y = target_engineer.align_features_and_targets(features, targets)
    
    # Split data
    import config
    splitter = DataSplitter(config)
    splits = splitter.split_with_config(X, y)
    
    # Save splits
    splitter.save_splits(splits, PROCESSED_DATA_DIR)
    logger.info("All splits saved successfully")
