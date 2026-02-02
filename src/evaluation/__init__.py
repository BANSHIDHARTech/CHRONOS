"""
Evaluation Module

Contains metrics and evaluation utilities for the CHRONOS system.
"""

from .metrics import (
    calculate_per_regime_accuracy,
    calculate_confidence_calibration,
    calculate_quintile_confusion_matrix,
    calculate_regime_transition_accuracy,
    generate_evaluation_report
)

__all__ = [
    'calculate_per_regime_accuracy',
    'calculate_confidence_calibration',
    'calculate_quintile_confusion_matrix',
    'calculate_regime_transition_accuracy',
    'generate_evaluation_report'
]
