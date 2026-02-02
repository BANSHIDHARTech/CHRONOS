"""
CHRONOS Interpretability Module

SHAP-based model interpretation for regime-specialized XGBoost ensemble.
Provides global and local explanations, cross-regime feature importance
analysis, and financial theory validation.
"""

from .shap_analyzer import (
    SHAPAnalyzer,
    create_shap_directories,
    REGIME_NAMES,
    REGIME_COLORS
)
from .feature_importance import (
    FeatureImportanceAnalyzer,
    EXPECTED_TOP_FEATURES
)

__all__ = [
    'SHAPAnalyzer',
    'create_shap_directories',
    'FeatureImportanceAnalyzer',
    'REGIME_NAMES',
    'REGIME_COLORS',
    'EXPECTED_TOP_FEATURES'
]
