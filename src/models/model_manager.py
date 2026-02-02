"""
Model Persistence Manager

Utilities for saving and loading trained XGBoost ensemble models.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import xgboost as xgb

from .xgboost_ensemble import RegimeSpecializedEnsemble

logger = logging.getLogger(__name__)

REGIME_FILE_NAMES = {0: 'euphoria', 1: 'complacency', 2: 'capitulation'}


def save_ensemble(
    ensemble: RegimeSpecializedEnsemble,
    output_dir: str,
    feature_names: Optional[List[str]] = None,
    additional_metadata: Optional[Dict] = None
) -> Dict[str, str]:
    """Save the trained ensemble to disk."""
    logger.info(f"Saving ensemble to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    
    # Save each regime model
    for regime_idx in [0, 1, 2]:
        model_key = f'regime_{regime_idx}'
        model = ensemble.get_model(regime_idx)
        
        if model is not None:
            regime_name = REGIME_FILE_NAMES[regime_idx]
            filename = f'xgb_regime_{regime_idx}_{regime_name}.json'
            filepath = os.path.join(output_dir, filename)
            model.save_model(filepath)
            saved_paths[model_key] = filepath
            logger.info(f"Saved {model_key} to {filepath}")
    
    # Prepare and save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'hyperparameters': ensemble.get_hyperparameters(),
        'trained_models': list(ensemble.models.keys()),
        'feature_names': feature_names or ensemble.feature_names,
    }
    if additional_metadata:
        metadata.update(additional_metadata)
    
    metadata_path = os.path.join(output_dir, 'ensemble_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_paths['metadata'] = metadata_path
    
    # Save training metrics
    if ensemble.training_metrics:
        metrics_path = os.path.join(output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(ensemble.training_metrics, f, indent=2)
        saved_paths['training_metrics'] = metrics_path
    
    return saved_paths


def load_ensemble(
    model_dir: str,
    validate_features: Optional[List[str]] = None
) -> RegimeSpecializedEnsemble:
    """Load a trained ensemble from disk."""
    logger.info(f"Loading ensemble from {model_dir}")
    
    metadata_path = os.path.join(model_dir, 'ensemble_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Validate features if provided
    stored_features = metadata.get('feature_names')
    if validate_features and stored_features:
        if set(validate_features) != set(stored_features):
            raise ValueError("Feature names mismatch")
    
    # Create ensemble
    hyperparams = metadata.get('hyperparameters', {})
    ensemble = RegimeSpecializedEnsemble(**hyperparams)
    ensemble.feature_names = stored_features
    
    # Load regime models
    for regime_idx in [0, 1, 2]:
        model_key = f'regime_{regime_idx}'
        regime_name = REGIME_FILE_NAMES[regime_idx]
        filename = f'xgb_regime_{regime_idx}_{regime_name}.json'
        filepath = os.path.join(model_dir, filename)
        
        if os.path.exists(filepath):
            model = xgb.XGBClassifier()
            model.load_model(filepath)
            ensemble.models[model_key] = model
            logger.info(f"Loaded {model_key}")
    
    # Load training metrics
    metrics_path = os.path.join(model_dir, 'training_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            ensemble.training_metrics = json.load(f)
    
    return ensemble


def get_model_info(model_dir: str) -> Dict[str, Any]:
    """Get information about a saved ensemble."""
    metadata_path = os.path.join(model_dir, 'ensemble_metadata.json')
    if not os.path.exists(metadata_path):
        return {'error': 'Metadata not found'}
    
    with open(metadata_path, 'r') as f:
        return json.load(f)
