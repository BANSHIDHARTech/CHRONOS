"""
CHRONOS Data Validation Script

Validates that the backtest data pipeline is correctly configured:
- Features and returns alignment
- Required columns present
- Test period has sufficient data
- Feature names match ensemble expectations
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json


def validate_backtest_data():
    """Validate all data inputs for backtesting."""
    print("=" * 60)
    print("CHRONOS DATA VALIDATION")
    print("=" * 60)
    print()
    
    errors = []
    warnings = []
    
    # 1. Load features
    print("[1] Loading features.csv...")
    features_path = 'data/processed/features.csv'
    if not os.path.exists(features_path):
        errors.append(f"Features file not found: {features_path}")
        return errors, warnings
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    print(f"    [OK] Loaded {len(features_df)} rows, {len(features_df.columns)} columns")
    
    # 2. Load returns
    print("[2] Loading returns.csv...")
    returns_path = 'data/processed/returns.csv'
    if not os.path.exists(returns_path):
        errors.append(f"Returns file not found: {returns_path}")
        return errors, warnings
    
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    print(f"    [OK] Loaded {len(returns_df)} rows, {len(returns_df.columns)} columns")
    
    # 3. Check date overlap
    print("[3] Checking date alignment...")
    test_start = '2024-01-01'
    test_end = '2024-12-31'
    
    test_features = features_df.loc[test_start:test_end]
    test_returns = returns_df.loc[test_start:test_end]
    
    print(f"    Test period features: {len(test_features)} rows")
    print(f"    Test period returns: {len(test_returns)} rows")
    
    if len(test_features) < 40:
        errors.append(f"Insufficient test data: {len(test_features)} weeks (need >= 40)")
    else:
        print(f"    [OK] Sufficient test data ({len(test_features)} >= 40 weeks)")
    
    # 4. Check required asset columns in returns
    print("[4] Checking asset columns...")
    required_assets = ['SPY', 'TLT', 'GLD']
    missing_assets = [a for a in required_assets if a not in returns_df.columns]
    
    if missing_assets:
        errors.append(f"Missing asset columns in returns: {missing_assets}")
    else:
        print(f"    [OK] All required assets present: {required_assets}")
    
    # 5. Load and validate ensemble metadata
    print("[5] Validating ensemble metadata...")
    metadata_path = 'models/ensemble_metadata.json'
    
    if not os.path.exists(metadata_path):
        errors.append(f"Ensemble metadata not found: {metadata_path}")
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata.get('feature_names', [])
        print(f"    Ensemble expects {len(feature_names)} features")
        
        # Check if ensemble features exist in features_df
        missing_features = [f for f in feature_names if f not in features_df.columns]
        if missing_features:
            warnings.append(f"Some ensemble features missing in data: {missing_features[:5]}...")
        else:
            print(f"    [OK] All {len(feature_names)} ensemble features found in data")
    
    # 6. Check for missing values
    print("[6] Checking for missing values...")
    missing_counts = features_df[test_start:test_end].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        warnings.append(f"Columns with missing values: {list(cols_with_missing.index)}")
        print(f"    [WARN] {len(cols_with_missing)} columns have missing values")
    else:
        print("    [OK] No missing values in test period")
    
    # 7. Check models exist
    print("[7] Checking model files...")
    model_files = [
        'models/regime_detector.pkl',
        'models/xgb_regime_0_euphoria.json',
        'models/xgb_regime_1_complacency.json',
        'models/xgb_regime_2_capitulation.json'
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            errors.append(f"Model file not found: {model_file}")
        else:
            print(f"    [OK] {os.path.basename(model_file)}")
    
    # Summary
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if errors:
        print(f"[FAIL] {len(errors)} errors found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[OK] No errors found")
    
    if warnings:
        print(f"[WARN] {len(warnings)} warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[OK] No warnings")
    
    print()
    if not errors:
        print("[OK] Data validation PASSED - ready for backtesting!")
    else:
        print("[FAIL] Data validation FAILED - please fix errors above")
    
    return errors, warnings


if __name__ == '__main__':
    errors, warnings = validate_backtest_data()
    sys.exit(1 if errors else 0)
