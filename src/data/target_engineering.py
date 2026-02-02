"""
Target Engineering Module

Implements forward return computation and quintile-based classification.
IMPORTANT: Target variables are NOT shifted (they represent future returns).
"""

import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TargetEngineer:
    """
    Target engineering for quintile-based classification.
    
    CRITICAL: Unlike features, targets are NOT shifted because they represent
    future returns that we're trying to predict.
    """
    
    def __init__(self, config):
        """
        Initialize TargetEngineer with configuration.
        
        Args:
            config: Configuration module or object
        """
        self.config = config
    
    def compute_forward_returns(
        self,
        close_prices: pd.Series,
        days: int = 5
    ) -> pd.Series:
        """
        Compute forward returns (NOT shifted).
        
        This calculates the return from today's close to close in 'days' periods.
        
        Args:
            close_prices: Series of closing prices
            days: Number of days forward to compute returns
            
        Returns:
            Forward returns (NOT SHIFTED - this is the target variable)
        """
        # Target variable - future returns, not shifted
        # Formula: (price[t+days] - price[t]) / price[t]
        forward_returns = (close_prices.shift(-days) - close_prices) / close_prices
        
        return forward_returns
    
    def create_quintile_labels(
        self,
        forward_returns: pd.Series,
        labels: list = [1, 2, 3, 4, 5]
    ) -> pd.Series:
        """
        Create quintile labels from forward returns.
        
        Divides returns into 5 equal-sized buckets:
        - 1: Bottom 20% (worst performers)
        - 2: 20-40%
        - 3: 40-60% (middle)
        - 4: 60-80%
        - 5: Top 20% (best performers)
        
        Args:
            forward_returns: Series of forward returns
            labels: Quintile labels (default [1,2,3,4,5])
            
        Returns:
            Series of quintile labels
        """
        try:
            # Drop NaN values before creating quintiles
            valid_returns = forward_returns.dropna()
            
            # Use qcut to divide into quintiles
            quintiles_valid = pd.qcut(
                valid_returns,
                q=5,
                labels=labels,
                duplicates='drop'  # Handle edge cases with identical boundaries
            )
            
            # Create full series with NaN for invalid indices
            quintiles = pd.Series(index=forward_returns.index, dtype='Int64')
            quintiles.loc[valid_returns.index] = quintiles_valid.astype(int)
            
            logger.info("Quintile distribution:")
            logger.info(quintiles.value_counts().sort_index())
            
            return quintiles
            
        except ValueError as e:
            logger.error(f"Error creating quintiles: {e}")
            logger.warning("This may occur if returns have very low variance")
            
            # Fallback: use rank-based percentiles
            logger.info("Using fallback method: rank-based percentiles")
            percentile_rank = forward_returns.rank(pct=True)
            
            # Manually assign quintiles based on percentile
            quintiles = pd.cut(
                percentile_rank,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=labels,
                include_lowest=True
            )
            
            return quintiles.astype(int)
    
    def engineer_targets(
        self,
        close_prices: pd.Series,
        forward_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Engineer target variables from closing prices.
        
        Args:
            close_prices: Series of closing prices
            forward_days: Number of days forward (uses config if None)
            
        Returns:
            DataFrame with columns ['forward_returns', 'quintile_label']
        """
        if forward_days is None:
            forward_days = self.config.FORWARD_RETURN_DAYS
        
        logger.info(f"Engineering targets with {forward_days}-day forward returns...")
        
        # Compute forward returns
        forward_returns = self.compute_forward_returns(close_prices, days=forward_days)
        
        # Create quintile labels
        quintile_labels = self.create_quintile_labels(
            forward_returns,
            labels=self.config.QUINTILE_LABELS
        )
        
        # Create target DataFrame
        targets_df = pd.DataFrame({
            'forward_returns': forward_returns,
            'quintile_label': quintile_labels
        }, index=close_prices.index)
        
        # Drop rows where forward returns are NaN (last forward_days rows)
        initial_rows = len(targets_df)
        targets_df = targets_df.dropna()
        dropped_rows = initial_rows - len(targets_df)
        
        logger.info(f"Dropped {dropped_rows} rows (insufficient forward data)")
        logger.info(f"Final target shape: {targets_df.shape}")
        
        # Log statistics
        logger.info("\nTarget statistics:")
        logger.info(f"Forward returns mean: {targets_df['forward_returns'].mean():.4f}")
        logger.info(f"Forward returns std: {targets_df['forward_returns'].std():.4f}")
        logger.info(f"Forward returns min: {targets_df['forward_returns'].min():.4f}")
        logger.info(f"Forward returns max: {targets_df['forward_returns'].max():.4f}")
        
        return targets_df
    
    def align_features_and_targets(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align features and targets on matching indices.
        
        Args:
            features_df: DataFrame of engineered features
            targets_df: DataFrame of target variables
            
        Returns:
            Tuple of (X, y) where:
            - X: Aligned features DataFrame
            - y: Aligned quintile labels Series
        """
        logger.info("Aligning features and targets...")
        
        # Inner join to keep only matching dates
        aligned = features_df.join(targets_df, how='inner')
        
        # Verify alignment
        if len(aligned) == 0:
            raise ValueError("No overlapping dates between features and targets")
        
        # Split back into X and y
        X = aligned[features_df.columns]
        y = aligned['quintile_label']
        
        logger.info(f"Aligned data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Date range: {X.index[0]} to {X.index[-1]}")
        
        # Log class distribution
        logger.info("\nClass distribution after alignment:")
        class_counts = y.value_counts().sort_index()
        for quintile, count in class_counts.items():
            pct = (count / len(y)) * 100
            logger.info(f"  Quintile {quintile}: {count} samples ({pct:.1f}%)")
        
        # Check for class imbalance
        expected_pct = 20.0
        for quintile, count in class_counts.items():
            actual_pct = (count / len(y)) * 100
            diff = abs(actual_pct - expected_pct)
            
            if diff > 5:  # More than 5% deviation
                logger.warning(
                    f"Class imbalance detected for quintile {quintile}: "
                    f"{actual_pct:.1f}% (expected ~20%)"
                )
        
        # Verify no NaN values
        if X.isnull().any().any():
            logger.warning("NaN values detected in features after alignment")
            nan_cols = X.columns[X.isnull().any()].tolist()
            logger.warning(f"Columns with NaN: {nan_cols}")
        
        if y.isnull().any():
            logger.warning("NaN values detected in targets after alignment")
        
        return X, y


# Example usage
if __name__ == '__main__':
    from config import (
        TICKERS, DATA_START_DATE, DATA_END_DATE,
        FEATURES_FILE, TARGETS_FILE, ALIGNED_DATA_FILE
    )
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
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
    
    # Align features and targets
    X, y = target_engineer.align_features_and_targets(features, targets)
    
    # Save
    import os
    os.makedirs(os.path.dirname(TARGETS_FILE), exist_ok=True)
    
    targets.to_pickle(TARGETS_FILE)
    logger.info(f"Targets saved to {TARGETS_FILE}")
    
    aligned_data = pd.concat([X, y], axis=1)
    aligned_data.to_pickle(ALIGNED_DATA_FILE)
    logger.info(f"Aligned data saved to {ALIGNED_DATA_FILE}")
