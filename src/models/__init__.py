"""Models module for regime detection and prediction"""

from src.models.regime_detector import RegimeDetector
from src.models.hmm_optimizer import (
    select_optimal_states,
    plot_bic_curve,
    compare_covariance_types,
    cross_validate_hmm
)
from src.models.xgboost_ensemble import RegimeSpecializedEnsemble
from src.models.predictor import AdaptivePredictor
from src.models.model_manager import save_ensemble, load_ensemble

__all__ = [
    'RegimeDetector',
    'select_optimal_states',
    'plot_bic_curve',
    'compare_covariance_types',
    'cross_validate_hmm',
    'RegimeSpecializedEnsemble',
    'AdaptivePredictor',
    'save_ensemble',
    'load_ensemble'
]

