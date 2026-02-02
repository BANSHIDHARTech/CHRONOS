"""
SHAP Analysis Runner

End-to-end script for running comprehensive SHAP analysis on trained ensemble.
Generates all visualizations and reports for model interpretability.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.models.model_manager import load_ensemble
from src.models.regime_detector import RegimeDetector
from src.interpretability.shap_analyzer import SHAPAnalyzer, create_shap_directories, REGIME_NAMES
from src.interpretability.feature_importance import FeatureImportanceAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_and_models() -> Dict[str, Any]:
    """
    Load processed data and trained models.
    
    Returns:
        Dictionary with features, returns, regime_labels, ensemble, feature_names
    """
    logger.info("=" * 60)
    logger.info("LOADING DATA AND MODELS")
    logger.info("=" * 60)
    
    # Load ensemble first to get feature names
    ensemble = load_ensemble(config.MODELS_DIR)
    logger.info("Loaded ensemble model")
    
    # Get feature names from ensemble
    if hasattr(ensemble, 'feature_names') and ensemble.feature_names:
        feature_names = ensemble.feature_names
    else:
        # Try to load from metadata
        import json
        metadata_path = os.path.join(config.MODELS_DIR, 'ensemble_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_names = metadata.get('feature_names', [])
    
    logger.info(f"Feature names ({len(feature_names)}): {feature_names[:5]}...")
    
    # Load regime labels
    regime_labels_path = os.path.join(config.PROCESSED_DATA_DIR, 'regime_labels.csv')
    if os.path.exists(regime_labels_path):
        regime_labels = pd.read_csv(regime_labels_path)
        # Convert date column to string for matching
        regime_labels['date_str'] = pd.to_datetime(regime_labels['date']).dt.strftime('%Y-%m-%d')
        regime_labels.set_index('date_str', inplace=True)
        logger.info(f"Loaded regime labels: {len(regime_labels)} rows")
    else:
        raise FileNotFoundError(f"Regime labels not found at {regime_labels_path}")
    
    # Load raw data and engineer features for full training period
    # This ensures we have data that aligns with regime labels
    from src.data.data_loader import DataLoader
    from src.data.feature_engineering import FeatureEngineer
    
    logger.info("Loading raw data and engineering features...")
    loader = DataLoader(raw_data_dir=config.RAW_DATA_DIR)
    
    tickers = list(config.TICKERS.keys())
    raw_data = loader.get_data(
        tickers=tickers,
        start_date=config.TRAIN_START,  # Use training period
        end_date=config.TRAIN_END,
        use_cache=True
    )
    
    engineer = FeatureEngineer(config)
    features_df = engineer.engineer_features(raw_data, primary_ticker='SPY')
    
    # Convert index to string for matching
    features_df.index = pd.to_datetime(features_df.index).strftime('%Y-%m-%d')
    
    logger.info(f"Engineered features: {features_df.shape}")
    
    return {
        'features_df': features_df,
        'regime_labels': regime_labels,
        'ensemble': ensemble,
        'feature_names': feature_names
    }


def split_by_regime(
    features_df: pd.DataFrame,
    regime_labels: pd.DataFrame,
    feature_names: List[str]
) -> Dict[int, np.ndarray]:
    """
    Split feature data by regime.
    
    Args:
        features_df: Full feature DataFrame (index is date string)
        regime_labels: DataFrame with regime_id column (index is date string)
        feature_names: List of feature names to use
        
    Returns:
        Dictionary mapping regime indices to feature matrices
    """
    X_dict = {}
    
    # Find common dates between features and regime labels
    common_dates = features_df.index.intersection(regime_labels.index)
    logger.info(f"Found {len(common_dates)} dates with both features and regime labels")
    
    if len(common_dates) == 0:
        logger.error("No overlapping dates between features and regime labels!")
        logger.error(f"Feature dates sample: {list(features_df.index[:5])}")
        logger.error(f"Regime dates sample: {list(regime_labels.index[:5])}")
        return X_dict
    
    # Filter to common dates
    features_aligned = features_df.loc[common_dates]
    labels_aligned = regime_labels.loc[common_dates]
    
    # Filter to available feature columns
    available_cols = [c for c in feature_names if c in features_aligned.columns]
    if len(available_cols) < len(feature_names):
        missing = set(feature_names) - set(available_cols)
        logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}")
    
    logger.info(f"Using {len(available_cols)} features")
    
    for regime_idx in range(3):
        # Get dates for this regime
        regime_mask = labels_aligned['regime_id'] == regime_idx
        regime_dates = labels_aligned[regime_mask].index
        
        # Filter features to matching dates
        X_regime = features_aligned.loc[regime_dates, available_cols]
        X_regime = X_regime.dropna()
        
        if len(X_regime) > 0:
            X_dict[regime_idx] = X_regime.values
            logger.info(f"Regime {regime_idx} ({REGIME_NAMES[regime_idx]}): {len(X_regime)} samples")
        else:
            logger.warning(f"No samples for regime {regime_idx}")
    
    return X_dict


def run_shap_analysis(
    data_split: str = 'all',
    regimes: str = 'all',
    n_samples: int = 5,
    output_dir: str = 'outputs/shap'
) -> Dict[str, Any]:
    """
    Run complete SHAP analysis pipeline.
    
    Args:
        data_split: Which data split to analyze ('train', 'val', 'test', 'all')
        regimes: Which regimes to analyze ('0', '1', '2', 'all')
        n_samples: Number of waterfall samples per regime
        output_dir: Base output directory
        
    Returns:
        Dictionary with analysis results and file paths
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("CHRONOS SHAP INTERPRETABILITY ANALYSIS")
    logger.info("=" * 60)
    
    # Create directories
    directories = create_shap_directories(output_dir)
    
    # Load data and models
    data = load_data_and_models()
    features_df = data['features_df']
    regime_labels = data['regime_labels']
    ensemble = data['ensemble']
    feature_names = data['feature_names']
    
    # Determine which regimes to analyze
    if regimes == 'all':
        regime_indices = [0, 1, 2]
    else:
        regime_indices = [int(r) for r in regimes.split(',')]
    
    # Split data by regime
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING DATA BY REGIME")
    logger.info("=" * 60)
    
    X_dict = split_by_regime(features_df, regime_labels, feature_names)
    
    # Initialize SHAP analyzer
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING SHAP ANALYZER")
    logger.info("=" * 60)
    
    shap_analyzer = SHAPAnalyzer(
        ensemble=ensemble,
        feature_names=feature_names,
        output_dir=output_dir
    )
    
    # Generate plots for each regime
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING SHAP PLOTS")
    logger.info("=" * 60)
    
    all_paths = {}
    
    for regime_idx in regime_indices:
        if regime_idx not in X_dict:
            logger.warning(f"Skipping regime {regime_idx} - no data")
            continue
        
        logger.info(f"\n--- Regime {regime_idx}: {REGIME_NAMES[regime_idx]} ---")
        
        X = X_dict[regime_idx]
        
        # Generate all plots for this regime
        paths = shap_analyzer.generate_all_plots(
            regime_idx=regime_idx,
            X=X,
            n_waterfall_samples=n_samples
        )
        
        all_paths[f'regime_{regime_idx}'] = paths
    
    # Cross-regime analysis - only if we have data
    if not X_dict:
        logger.error("No regime data available! Cannot perform cross-regime analysis.")
        return {
            'paths': all_paths,
            'validation': {},
            'importance_df': None,
            'elapsed_time': time.time() - start_time
        }
    
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-REGIME FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)
    
    fi_analyzer = FeatureImportanceAnalyzer(
        shap_analyzer=shap_analyzer,
        output_dir=os.path.join(output_dir, 'cross_regime')
    )
    
    # Compute cross-regime importance
    fi_analyzer.compute_cross_regime_importance(X_dict)
    
    # Only generate outputs if we have importance data
    if fi_analyzer.importance_df is not None and len(fi_analyzer.importance_df) > 0:
        # Generate all cross-regime outputs
        cross_regime_paths = fi_analyzer.generate_all_outputs()
        all_paths['cross_regime'] = cross_regime_paths
        
        # Validate financial theory
        logger.info("\n" + "=" * 60)
        logger.info("FINANCIAL THEORY VALIDATION")
        logger.info("=" * 60)
        
        validation_results = fi_analyzer.validate_financial_theory()
    else:
        logger.warning("No importance data computed, skipping cross-regime outputs")
        cross_regime_paths = {}
        validation_results = {}
    
    # Summary
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("SHAP ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Output directory: {output_dir}")
    
    # Log file counts
    file_count = 0
    for regime_key, paths in all_paths.items():
        if isinstance(paths, dict):
            for path_type, path_val in paths.items():
                if isinstance(path_val, list):
                    file_count += len(path_val)
                else:
                    file_count += 1
    
    logger.info(f"Generated {file_count} output files")
    
    # Print feature importance summary
    if fi_analyzer.importance_df is not None and len(fi_analyzer.importance_df) > 0:
        logger.info("\n--- Top 5 Features by Overall Importance ---")
        top_features = fi_analyzer.importance_df.head(5)
        for feature in top_features.index:
            importance = fi_analyzer.importance_df.loc[feature, 'overall_importance']
            logger.info(f"  {feature}: {importance:.4f}")
    
    # Save analysis log
    log_path = os.path.join(output_dir, 'analysis_log.txt')
    # Note: logging to file is handled by standard logging config
    
    return {
        'paths': all_paths,
        'validation': validation_results,
        'importance_df': fi_analyzer.importance_df,
        'elapsed_time': elapsed
    }


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Run SHAP analysis on CHRONOS ensemble')
    parser.add_argument(
        '--data-split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Data split to analyze'
    )
    parser.add_argument(
        '--regimes',
        type=str,
        default='all',
        help='Regimes to analyze (0, 1, 2, or "all")'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of waterfall samples per regime'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/shap',
        help='Output directory for SHAP analysis'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_shap_analysis(
            data_split=args.data_split,
            regimes=args.regimes,
            n_samples=args.n_samples,
            output_dir=args.output_dir
        )
        
        logger.info("\n[OK] SHAP analysis completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()
