"""
Validation Utilities

Anti-leakage validators and data integrity checks.
These utilities help ensure no future information leaks into training data.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_feature_shift(
    feature_series: pd.Series,
    original_series: pd.Series,
    feature_name: str
) -> bool:
    """
    Verify that a feature is properly shifted relative to original data.
    
    Args:
        feature_series: Computed feature series
        original_series: Original price/data series
        feature_name: Name of the feature for error messages
        
    Returns:
        True if properly shifted
        
    Raises:
        ValueError: If feature is not properly shifted
    """
    logger.info(f"Checking shift for feature: {feature_name}")
    
    # Get first valid indices
    feature_first_valid = feature_series.first_valid_index()
    original_first_valid = original_series.first_valid_index()
    
    if feature_first_valid is None:
        logger.warning(f"Feature {feature_name} has no valid values")
        return True
    
    if original_first_valid is None:
        raise ValueError("Original series has no valid values")
    
    # Feature should start later than original (due to shift)
    if feature_first_valid <= original_first_valid:
        raise ValueError(
            f"Feature {feature_name} is not properly shifted. "
            f"Feature starts at {feature_first_valid}, "
            f"original starts at {original_first_valid}. "
            f"Feature should start AFTER original data."
        )
    
    # Check that indices are aligned
    if not feature_series.index.equals(original_series.index):
        logger.warning(
            f"Feature {feature_name} and original series have different indices"
        )
    
    logger.info(f"✓ Feature {feature_name} is properly shifted")
    return True


def validate_rolling_windows(
    df: pd.DataFrame,
    min_periods_required: bool = True
) -> Dict[str, List[str]]:
    """
    Validate that rolling operations use explicit min_periods.
    
    This is a static check - in practice, developers should ensure
    all rolling operations in their code specify min_periods.
    
    Args:
        df: DataFrame to check
        min_periods_required: Whether to enforce min_periods requirement
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating rolling window operations...")
    
    results = {
        'status': 'passed',
        'warnings': [],
        'errors': []
    }
    
    # Check for columns that might be rolling window results
    # (This is a runtime check of the data, not the code)
    
    for col in df.columns:
        # Check for initial NaN values (common in rolling windows)
        first_valid_idx = df[col].first_valid_index()
        
        if first_valid_idx is not None:
            nan_count_before = df.loc[:first_valid_idx, col].isna().sum()
            
            if nan_count_before > 0:
                results['warnings'].append(
                    f"Column {col} has {nan_count_before} leading NaN values "
                    f"(typical for rolling windows)"
                )
    
    if results['warnings']:
        logger.info(f"Found {len(results['warnings'])} potential rolling window columns")
        logger.info(
            "Ensure all rolling operations in code use explicit min_periods parameter"
        )
    else:
        logger.info("✓ No obvious rolling window issues detected")
    
    return results


def check_data_integrity(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data integrity checks.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with integrity check results
    """
    logger.info("Running data integrity checks...")
    
    issues = []
    statistics = {}
    
    # Check 1: Datetime index is sorted
    if not df.index.is_monotonic_increasing:
        issues.append("Index is not sorted in ascending order")
    else:
        logger.info("✓ Index is properly sorted")
    
    # Check 2: No duplicate indices
    duplicates = df.index.duplicated()
    if duplicates.any():
        num_duplicates = duplicates.sum()
        issues.append(f"Found {num_duplicates} duplicate index entries")
        statistics['duplicate_dates'] = df.index[duplicates].tolist()
    else:
        logger.info("✓ No duplicate indices found")
    
    # Check 3: Check for gaps in time series
    if isinstance(df.index, pd.DatetimeIndex):
        date_diffs = df.index.to_series().diff()
        max_gap = date_diffs.max()
        
        statistics['max_gap_days'] = max_gap.days if pd.notna(max_gap) else 0
        
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=10)]
        if not large_gaps.empty:
            issues.append(
                f"Found {len(large_gaps)} gaps larger than 10 days in time series"
            )
            statistics['large_gaps'] = large_gaps.index.tolist()
        else:
            logger.info("✓ No large gaps in time series")
    
    # Check 4: Missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        statistics['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Calculate percentage
        missing_pct = (missing_counts / len(df) * 100)
        high_missing = missing_pct[missing_pct > 5]
        
        if not high_missing.empty:
            issues.append(
                f"High missing value percentage (>5%) in {len(high_missing)} columns"
            )
    else:
        logger.info("✓ No missing values detected")
    
    # Check 5: Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = df[numeric_cols].isin([np.inf, -np.inf]).sum()
    total_inf = inf_counts.sum()
    
    if total_inf > 0:
        issues.append(f"Found {total_inf} infinite values")
        statistics['infinite_values'] = inf_counts[inf_counts > 0].to_dict()
    else:
        logger.info("✓ No infinite values detected")
    
    # Check 6: Data type consistency
    statistics['dtypes'] = df.dtypes.value_counts().to_dict()
    
    # Summary
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("✓ All data integrity checks passed")
    else:
        logger.warning(f"Data integrity issues found: {len(issues)}")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'statistics': statistics
    }


def validate_train_test_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_name: str = "train/test"
) -> bool:
    """
    Validate that train and test splits have no overlap and proper ordering.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        split_name: Name of the split for error messages
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    logger.info(f"Validating {split_name} split...")
    
    # Check that both DataFrames have data
    if len(train_df) == 0:
        raise ValueError(f"Training set is empty")
    
    if len(test_df) == 0:
        raise ValueError(f"Test set is empty")
    
    # Check date ranges
    if isinstance(train_df.index, pd.DatetimeIndex) and isinstance(test_df.index, pd.DatetimeIndex):
        train_min = train_df.index.min()
        train_max = train_df.index.max()
        test_min = test_df.index.min()
        test_max = test_df.index.max()
        
        logger.info(f"Train range: {train_min} to {train_max}")
        logger.info(f"Test range: {test_min} to {test_max}")
        
        # Ensure test dates are strictly after train dates
        if test_min <= train_max:
            raise ValueError(
                f"Date overlap detected: "
                f"train_max={train_max}, test_min={test_min}. "
                f"Test dates must be strictly after training dates."
            )
        
        logger.info("✓ No date overlap - proper temporal ordering")
        
        # Check for large gap
        gap_days = (test_min - train_max).days
        if gap_days > 7:
            logger.warning(f"Large gap between train and test: {gap_days} days")
    
    # Check for index overlap (regardless of date type)
    index_overlap = train_df.index.intersection(test_df.index)
    if len(index_overlap) > 0:
        raise ValueError(
            f"Found {len(index_overlap)} overlapping indices between train and test"
        )
    
    logger.info("✓ No index overlap detected")
    
    # Check column alignment
    if not train_df.columns.equals(test_df.columns):
        logger.warning("Train and test have different columns")
        
        train_only = set(train_df.columns) - set(test_df.columns)
        test_only = set(test_df.columns) - set(train_df.columns)
        
        if train_only:
            logger.warning(f"Columns only in train: {train_only}")
        if test_only:
            logger.warning(f"Columns only in test: {test_only}")
    else:
        logger.info("✓ Column alignment validated")
    
    return True


def detect_lookahead_bias(
    features: pd.DataFrame,
    targets: pd.Series,
    suspicious_correlation_threshold: float = 0.95
) -> Dict:
    """
    Detect potential lookahead bias by checking for suspiciously high correlations.
    
    Args:
        features: Feature DataFrame
        targets: Target Series
        suspicious_correlation_threshold: Correlation threshold for flagging
        
    Returns:
        Dictionary with detection results
    """
    logger.info("Checking for potential lookahead bias...")
    
    results = {
        'suspicious_features': [],
        'correlations': {}
    }
    
    # For regression targets, check correlations
    if targets.dtype in [np.float64, np.float32]:
        for col in features.columns:
            # Skip non-numeric columns
            if features[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue
            
            # Calculate correlation
            corr = features[col].corr(targets)
            
            if pd.notna(corr):
                results['correlations'][col] = corr
                
                # Flag suspiciously high correlations
                if abs(corr) > suspicious_correlation_threshold:
                    results['suspicious_features'].append(col)
                    logger.warning(
                        f"Suspiciously high correlation for {col}: {corr:.4f}"
                    )
    
    if results['suspicious_features']:
        logger.warning(
            f"Found {len(results['suspicious_features'])} features with "
            f"suspiciously high correlation (>{suspicious_correlation_threshold})"
        )
        logger.warning("This may indicate lookahead bias - verify feature engineering")
    else:
        logger.info("✓ No suspicious correlations detected")
    
    return results


def validate_pipeline_integrity(
    raw_data: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: Dict
) -> bool:
    """
    Comprehensive pipeline integrity validation.
    
    Args:
        raw_data: Original raw data
        features: Engineered features
        targets: Target variables
        splits: Dictionary of train/val/test splits
        
    Returns:
        True if all validations pass
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE PIPELINE INTEGRITY VALIDATION")
    logger.info("=" * 80)
    
    all_passed = True
    
    # 1. Check raw data integrity
    logger.info("\n1. Validating raw data...")
    raw_check = check_data_integrity(raw_data)
    if not raw_check['is_valid']:
        logger.error("Raw data integrity check FAILED")
        all_passed = False
    
    # 2. Check features integrity
    logger.info("\n2. Validating features...")
    features_check = check_data_integrity(features)
    if not features_check['is_valid']:
        logger.error("Features integrity check FAILED")
        all_passed = False
    
    # 3. Validate splits
    logger.info("\n3. Validating splits...")
    try:
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        validate_train_test_split(X_train, X_val, "train/val")
        validate_train_test_split(X_val, X_test, "val/test")
        validate_train_test_split(X_train, X_test, "train/test")
        
        logger.info("✓ All split validations passed")
    except Exception as e:
        logger.error(f"Split validation FAILED: {e}")
        all_passed = False
    
    # 4. Check for lookahead bias
    logger.info("\n4. Checking for lookahead bias...")
    try:
        X_train, y_train = splits['train']
        lookahead_check = detect_lookahead_bias(X_train, y_train)
        
        if lookahead_check['suspicious_features']:
            logger.warning("Potential lookahead bias detected - review flagged features")
    except Exception as e:
        logger.error(f"Lookahead bias check failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✓ ALL PIPELINE INTEGRITY CHECKS PASSED")
    else:
        logger.error("✗ PIPELINE INTEGRITY CHECKS FAILED")
    logger.info("=" * 80)
    
    return all_passed


# Example usage
if __name__ == '__main__':
    # Create test data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Simulate price data
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(len(dates)) * 2),
        index=dates,
        name='Close'
    )
    
    # Simulate a properly shifted feature
    shifted_feature = prices.rolling(20).mean().shift(1)
    
    # Test shift validation
    try:
        check_feature_shift(shifted_feature, prices, 'SMA_20')
        print("✓ Shift validation passed")
    except ValueError as e:
        print(f"✗ Shift validation failed: {e}")
    
    # Test data integrity
    df = pd.DataFrame({
        'price': prices,
        'sma': shifted_feature
    })
    
    integrity = check_data_integrity(df)
    print(f"\nData integrity: {'PASSED' if integrity['is_valid'] else 'FAILED'}")
