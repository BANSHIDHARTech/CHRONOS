"""
Evaluation Metrics Module

Comprehensive evaluation functions for the CHRONOS regime-specialized
XGBoost ensemble, including per-regime accuracy, confidence calibration,
and transition analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

# Configure logging
logger = logging.getLogger(__name__)

# Regime name mapping
REGIME_NAMES = {
    0: 'Euphoria',
    1: 'Complacency',
    2: 'Capitulation'
}


def calculate_per_regime_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate accuracy separately for each regime.
    
    Args:
        y_true: True quintile labels
        y_pred: Predicted quintile labels
        regime_labels: Regime assignments for each sample
        
    Returns:
        Dictionary with per-regime and overall accuracy:
        {
            'regime_0': accuracy for Euphoria,
            'regime_1': accuracy for Complacency,
            'regime_2': accuracy for Capitulation,
            'overall': overall accuracy,
            'regime_0_name': 'Euphoria',
            ...
        }
    """
    results = {}
    
    # Overall accuracy
    results['overall'] = float(accuracy_score(y_true, y_pred))
    
    # Per-regime accuracy
    for regime_idx in [0, 1, 2]:
        mask = regime_labels == regime_idx
        n_samples = np.sum(mask)
        
        if n_samples > 0:
            regime_acc = float(accuracy_score(y_true[mask], y_pred[mask]))
            results[f'regime_{regime_idx}'] = regime_acc
            results[f'regime_{regime_idx}_samples'] = int(n_samples)
        else:
            results[f'regime_{regime_idx}'] = None
            results[f'regime_{regime_idx}_samples'] = 0
        
        results[f'regime_{regime_idx}_name'] = REGIME_NAMES[regime_idx]
    
    # Format regime accuracies, handling None values
    regime_0_str = f"{results.get('regime_0'):.4f}" if results.get('regime_0') is not None else "N/A"
    regime_1_str = f"{results.get('regime_1'):.4f}" if results.get('regime_1') is not None else "N/A"
    regime_2_str = f"{results.get('regime_2'):.4f}" if results.get('regime_2') is not None else "N/A"
    
    logger.info(
        f"Per-regime accuracy: "
        f"Euphoria={regime_0_str}, "
        f"Complacency={regime_1_str}, "
        f"Capitulation={regime_2_str}, "
        f"Overall={results['overall']:.4f}"
    )
    
    return results


def calculate_confidence_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
    bins: List[float] = None
) -> Dict[str, Any]:
    """
    Calculate confidence calibration curve.
    
    Measures how well predicted confidence matches actual accuracy
    across different confidence levels.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        y_pred: Predicted class labels
        bins: Bin edges for confidence levels (default: [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
    Returns:
        Dictionary containing:
            - bin_edges: Confidence bin edges
            - bin_centers: Center of each bin
            - bin_counts: Number of samples in each bin
            - mean_confidence: Mean predicted confidence per bin
            - actual_accuracy: Actual accuracy per bin
            - calibration_error: Absolute difference between confidence and accuracy
    """
    if bins is None:
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Get maximum confidence for each prediction
    max_confidence = np.max(y_pred_proba, axis=1)
    
    # Check if predictions are correct
    correct = (y_pred == y_true).astype(int)
    
    # Calculate statistics for each bin
    n_bins = len(bins) - 1
    bin_counts = np.zeros(n_bins)
    mean_confidence = np.zeros(n_bins)
    actual_accuracy = np.zeros(n_bins)
    calibration_error = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (max_confidence >= bins[i]) & (max_confidence < bins[i + 1])
        
        # Handle last bin inclusively
        if i == n_bins - 1:
            bin_mask = (max_confidence >= bins[i]) & (max_confidence <= bins[i + 1])
        
        count = np.sum(bin_mask)
        bin_counts[i] = count
        
        if count > 0:
            mean_confidence[i] = np.mean(max_confidence[bin_mask])
            actual_accuracy[i] = np.mean(correct[bin_mask])
            calibration_error[i] = np.abs(mean_confidence[i] - actual_accuracy[i])
        else:
            mean_confidence[i] = np.nan
            actual_accuracy[i] = np.nan
            calibration_error[i] = np.nan
    
    # Calculate bin centers
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]
    
    # Calculate Expected Calibration Error (ECE)
    valid_mask = ~np.isnan(calibration_error)
    if np.sum(valid_mask) > 0:
        ece = np.sum(bin_counts[valid_mask] * calibration_error[valid_mask]) / np.sum(bin_counts[valid_mask])
    else:
        ece = np.nan
    
    return {
        'bin_edges': bins,
        'bin_centers': bin_centers,
        'bin_counts': bin_counts.tolist(),
        'mean_confidence': mean_confidence.tolist(),
        'actual_accuracy': actual_accuracy.tolist(),
        'calibration_error': calibration_error.tolist(),
        'expected_calibration_error': float(ece) if not np.isnan(ece) else None
    }


def calculate_quintile_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Generate confusion matrix and classification metrics for quintile predictions.
    
    Args:
        y_true: True quintile labels (1-5)
        y_pred: Predicted quintile labels (1-5)
        
    Returns:
        Dictionary containing:
            - confusion_matrix: 5x5 confusion matrix
            - labels: Quintile labels [1, 2, 3, 4, 5]
            - per_class_precision: Precision per quintile
            - per_class_recall: Recall per quintile
            - per_class_f1: F1-score per quintile
            - classification_report: Full text classification report
    """
    labels = [1, 2, 3, 4, 5]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Generate text report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    
    return {
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'classification_report': report
    }


def calculate_regime_transition_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: np.ndarray,
    transition_window: int = 5
) -> Dict[str, float]:
    """
    Measure prediction accuracy during regime transitions vs stable periods.
    
    Regime transitions are often the most challenging periods for prediction.
    This metric helps identify if the model struggles during market regime changes.
    
    Args:
        y_true: True quintile labels
        y_pred: Predicted quintile labels
        regime_labels: Regime assignments for each sample
        transition_window: Number of periods around a transition to consider
        
    Returns:
        Dictionary containing:
            - stable_accuracy: Accuracy during stable regime periods
            - transition_accuracy: Accuracy during regime transitions
            - stable_samples: Number of stable period samples
            - transition_samples: Number of transition period samples
            - n_transitions: Number of regime transitions detected
    """
    n_samples = len(regime_labels)
    
    # Identify transition points (where regime changes)
    transition_mask = np.zeros(n_samples, dtype=bool)
    transition_points = []
    
    for i in range(1, n_samples):
        if regime_labels[i] != regime_labels[i - 1]:
            transition_points.append(i)
            # Mark samples within window of transition
            start = max(0, i - transition_window)
            end = min(n_samples, i + transition_window + 1)
            transition_mask[start:end] = True
    
    stable_mask = ~transition_mask
    
    # Calculate accuracies
    n_stable = np.sum(stable_mask)
    n_transition = np.sum(transition_mask)
    
    stable_accuracy = None
    transition_accuracy = None
    
    if n_stable > 0:
        stable_accuracy = float(accuracy_score(y_true[stable_mask], y_pred[stable_mask]))
    
    if n_transition > 0:
        transition_accuracy = float(accuracy_score(y_true[transition_mask], y_pred[transition_mask]))
    
    results = {
        'stable_accuracy': stable_accuracy,
        'transition_accuracy': transition_accuracy,
        'stable_samples': int(n_stable),
        'transition_samples': int(n_transition),
        'n_transitions': len(transition_points),
        'transition_points': transition_points
    }
    
    # Format accuracies handling None values
    stable_str = f"{stable_accuracy:.4f}" if stable_accuracy is not None else "N/A"
    transition_str = f"{transition_accuracy:.4f}" if transition_accuracy is not None else "N/A"
    
    logger.info(
        f"Transition analysis: {len(transition_points)} transitions detected. "
        f"Stable accuracy: {stable_str}, "
        f"Transition accuracy: {transition_str}"
    )
    
    return results


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    regime_labels: np.ndarray,
    confidence_bins: List[float] = None,
    transition_window: int = 5,
    include_raw_predictions: bool = False
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report aggregating all metrics.
    
    Args:
        y_true: True quintile labels
        y_pred: Predicted quintile labels
        y_pred_proba: Predicted probabilities
        regime_labels: Regime assignments
        confidence_bins: Bins for calibration analysis
        transition_window: Window size for transition analysis
        include_raw_predictions: Whether to include raw predictions in report
        
    Returns:
        Comprehensive dictionary containing all evaluation metrics,
        suitable for JSON export
    """
    logger.info("Generating comprehensive evaluation report")
    
    report = {
        'summary': {},
        'per_regime_accuracy': {},
        'confusion_matrix': {},
        'calibration': {},
        'transition_analysis': {},
        'metadata': {}
    }
    
    # Per-regime accuracy
    per_regime = calculate_per_regime_accuracy(y_true, y_pred, regime_labels)
    report['per_regime_accuracy'] = per_regime
    
    # Confusion matrix and classification metrics
    cm_results = calculate_quintile_confusion_matrix(y_true, y_pred)
    report['confusion_matrix'] = cm_results
    
    # Confidence calibration
    calibration = calculate_confidence_calibration(
        y_true, y_pred_proba, y_pred, bins=confidence_bins
    )
    report['calibration'] = calibration
    
    # Regime transition analysis
    transition = calculate_regime_transition_accuracy(
        y_true, y_pred, regime_labels, transition_window=transition_window
    )
    report['transition_analysis'] = transition
    
    # Summary statistics
    report['summary'] = {
        'overall_accuracy': per_regime['overall'],
        'best_regime': max(
            [(k, v) for k, v in per_regime.items() 
             if k.startswith('regime_') and not k.endswith('_name') 
             and not k.endswith('_samples') and v is not None],
            key=lambda x: x[1],
            default=('none', 0)
        )[0],
        'worst_regime': min(
            [(k, v) for k, v in per_regime.items() 
             if k.startswith('regime_') and not k.endswith('_name') 
             and not k.endswith('_samples') and v is not None],
            key=lambda x: x[1],
            default=('none', 1)
        )[0],
        'expected_calibration_error': calibration['expected_calibration_error'],
        'n_regime_transitions': transition['n_transitions'],
        'total_samples': len(y_true)
    }
    
    # Metadata
    report['metadata'] = {
        'n_samples': len(y_true),
        'n_quintiles': 5,
        'n_regimes': 3,
        'confidence_bins': confidence_bins or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'transition_window': transition_window
    }
    
    # Optionally include raw predictions
    if include_raw_predictions:
        report['raw_predictions'] = {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'regime_labels': regime_labels.tolist()
        }
    
    logger.info(
        f"Evaluation report generated: overall_accuracy={report['summary']['overall_accuracy']:.4f}"
    )
    
    return report


def compare_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    regime_labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models' performance.
    
    Args:
        y_true: True labels
        predictions_dict: Dictionary mapping model names to predictions
        regime_labels: Regime assignments
        
    Returns:
        Dictionary with per-model performance metrics
    """
    results = {}
    
    for model_name, y_pred in predictions_dict.items():
        per_regime = calculate_per_regime_accuracy(y_true, y_pred, regime_labels)
        results[model_name] = per_regime
    
    return results
