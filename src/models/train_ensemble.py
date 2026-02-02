"""
XGBoost Ensemble Training Pipeline

End-to-end training script for the regime-specialized XGBoost ensemble.
Loads data, detects regimes using trained HMM, trains specialized models,
and saves results.
"""

import sys
import os
import logging
import json
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.xgboost_ensemble import RegimeSpecializedEnsemble
from src.models.model_manager import save_ensemble
from src.models.regime_detector import RegimeDetector
from src.evaluation.metrics import generate_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data():
    """Load processed features and targets from disk."""
    logger.info("Loading processed data...")
    
    features_path = config.FEATURES_FILE
    targets_path = config.TARGETS_FILE
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    
    with open(features_path, 'rb') as f:
        features_data = pickle.load(f)
    with open(targets_path, 'rb') as f:
        targets_data = pickle.load(f)
    
    logger.info(f"Loaded features: {type(features_data)}")
    logger.info(f"Loaded targets: {type(targets_data)}")
    
    return features_data, targets_data


def load_regime_detector(model_path: str = None) -> RegimeDetector:
    """Load trained HMM regime detector."""
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'regime_detector.pkl')
    
    logger.info(f"Loading regime detector from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Regime detector not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        detector = pickle.load(f)
    
    return detector


def prepare_training_data(features_df, targets_df, regime_labels):
    """Prepare data for ensemble training."""
    # Align data
    common_idx = features_df.index.intersection(targets_df.index)
    
    X = features_df.loc[common_idx].values
    y = targets_df.loc[common_idx].values.flatten()
    regimes = regime_labels[:len(common_idx)]
    
    feature_names = features_df.columns.tolist()
    
    logger.info(f"Prepared data: X={X.shape}, y={y.shape}, regimes={regimes.shape}")
    
    return X, y, regimes, feature_names


def split_by_date(features_df, targets_df, regime_labels):
    """Split data into train/val/test by date."""
    train_mask = (features_df.index >= config.TRAIN_START) & (features_df.index <= config.TRAIN_END)
    val_mask = (features_df.index >= config.VAL_START) & (features_df.index <= config.VAL_END)
    test_mask = (features_df.index >= config.TEST_START) & (features_df.index <= config.TEST_END)
    
    splits = {}
    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        idx = features_df.index[mask]
        splits[name] = {
            'X': features_df.loc[idx].values,
            'y': targets_df.loc[idx].values.flatten() if idx.isin(targets_df.index).all() else None,
            'regimes': regime_labels[mask.values] if len(regime_labels) == len(features_df) else None,
            'index': idx
        }
        logger.info(f"{name}: {len(idx)} samples")
    
    return splits


def train_ensemble_pipeline():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("CHRONOS XGBoost Ensemble Training Pipeline")
    logger.info("=" * 60)
    
    # Load data
    features_data, targets_data = load_processed_data()
    
    # Handle different data formats
    if isinstance(features_data, pd.DataFrame):
        features_df = features_data
    elif isinstance(features_data, dict):
        features_df = features_data.get('features', pd.DataFrame(features_data))
    else:
        raise ValueError(f"Unexpected features format: {type(features_data)}")
    
    if isinstance(targets_data, pd.DataFrame):
        targets_df = targets_data
    elif isinstance(targets_data, dict):
        targets_df = pd.DataFrame(targets_data)
    else:
        targets_df = pd.Series(targets_data)
    
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Targets shape: {targets_df.shape}")
    
    # If targets has multiple columns, use the quintile_label column
    if isinstance(targets_df, pd.DataFrame) and targets_df.shape[1] > 1:
        logger.info(f"Targets columns: {targets_df.columns.tolist()}")
        # Look for quintile_label column (preferred) or quintile column
        if 'quintile_label' in targets_df.columns:
            targets_series = targets_df['quintile_label']
        elif 'quintile' in targets_df.columns:
            targets_series = targets_df['quintile']
        else:
            # Use first column as fallback
            targets_series = targets_df.iloc[:, 0]
        logger.info(f"Using target column: {targets_series.name}")
    else:
        targets_series = targets_df.iloc[:, 0] if isinstance(targets_df, pd.DataFrame) else targets_df
    
    # Load regime detector and predict regimes
    try:
        # Try to load from regime_labels.csv first (most reliable)
        regime_labels_path = os.path.join(config.PROCESSED_DATA_DIR, 'regime_labels.csv')
        if os.path.exists(regime_labels_path):
            logger.info(f"Loading regime labels from {regime_labels_path}")
            regime_df = pd.read_csv(regime_labels_path, parse_dates=['date'])
            regime_df = regime_df.set_index('date')
            # Make sure the index aligns with features_df
            regime_df = regime_df.reindex(features_df.index, method='ffill')
            regime_labels = regime_df['regime_id'].values
            logger.info(f"Loaded regime labels from CSV: {len(regime_labels)} samples")
            # Count non-NaN values
            valid_mask = ~np.isnan(regime_labels)
            if valid_mask.any():
                logger.info(f"Regime distribution: {np.bincount(regime_labels[valid_mask].astype(int))}")
            else:
                raise ValueError("All regime labels are NaN")
        else:
            # Fallback: load detector and predict
            detector = load_regime_detector()
            if hasattr(detector, 'predict'):
                regime_labels = detector.predict(features_df.values)
                logger.info(f"Predicted regimes using detector: {np.bincount(regime_labels)}")
            else:
                raise ValueError("Detector has no predict method")
    except Exception as e:
        logger.warning(f"Could not load regime labels: {e}")
        logger.info("Using dummy regime labels (all Complacency)")
        regime_labels = np.ones(len(features_df), dtype=int)
    
    # Align data - use common index between features and targets
    common_idx = features_df.index.intersection(targets_series.index)
    X = features_df.loc[common_idx].values
    y = targets_series.loc[common_idx].values
    
    # Align regime labels with common index
    if len(regime_labels) == len(features_df):
        # Map regime labels to common_idx positions
        regime_mask = features_df.index.isin(common_idx)
        regimes = regime_labels[regime_mask]
    else:
        regimes = np.ones(len(common_idx), dtype=int)
    
    feature_names = features_df.columns.tolist()
    
    # Split by date
    train_mask = pd.to_datetime(common_idx) <= pd.to_datetime(config.TRAIN_END)
    val_mask = (pd.to_datetime(common_idx) >= pd.to_datetime(config.VAL_START)) & \
               (pd.to_datetime(common_idx) <= pd.to_datetime(config.VAL_END))
    
    X_train, y_train = X[train_mask], y[train_mask]
    regimes_train = regimes[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    regimes_val = regimes[val_mask]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Get XGBoost config
    xgb_config = getattr(config, 'XGBOOST_CONFIG', {
        'max_depth': 3,
        'reg_lambda': 1,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'early_stopping_rounds': 10,
        'random_state': 42
    })
    
    # Create and train ensemble
    ensemble = RegimeSpecializedEnsemble(**xgb_config)
    training_metrics = ensemble.train_regime_models(
        X_train, y_train, regimes_train,
        feature_names=feature_names
    )
    
    # Validate on validation set
    logger.info("\nValidating on validation set...")
    if len(X_val) > 0:
        y_pred_val = np.zeros(len(X_val), dtype=int)
        y_proba_val = np.zeros((len(X_val), 5))
        
        for regime_idx in np.unique(regimes_val):
            mask = regimes_val == regime_idx
            if np.sum(mask) > 0 and ensemble.get_model(regime_idx) is not None:
                y_pred_val[mask] = ensemble.predict(X_val[mask], regime_idx)
                y_proba_val[mask] = ensemble.predict_proba(X_val[mask], regime_idx)
        
        # Generate evaluation report
        eval_report = generate_evaluation_report(
            y_val, y_pred_val, y_proba_val, regimes_val
        )
        logger.info(f"Validation accuracy: {eval_report['summary']['overall_accuracy']:.4f}")
    
    # Save ensemble
    output_dir = getattr(config, 'MODEL_PATHS', {}).get('ensemble_dir', config.MODELS_DIR)
    save_paths = save_ensemble(ensemble, output_dir, feature_names)
    logger.info(f"Saved ensemble to {output_dir}")
    
    # Save evaluation report
    if len(X_val) > 0:
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(eval_report, f, indent=2, default=str)
        logger.info(f"Saved evaluation report to {report_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    return ensemble, training_metrics


if __name__ == '__main__':
    train_ensemble_pipeline()
