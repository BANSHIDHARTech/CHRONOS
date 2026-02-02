"""
HMM Validation Module

Provides validation and testing utilities for the regime detector.
Includes regime persistence checks, distribution validation, and visualization.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.models.regime_detector import RegimeDetector
from src.utils.regime_utils import (
    calculate_regime_durations,
    get_regime_statistics,
    identify_regime_switches,
    REGIME_NAMES,
    REGIME_CHARACTERISTICS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_regime_detector(
    detector: RegimeDetector,
    X_val: np.ndarray,
    dates_val: Optional[pd.DatetimeIndex] = None,
    X_train: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Validate regime detector on validation set.
    
    Args:
        detector: Fitted RegimeDetector instance
        X_val: Validation feature matrix
        dates_val: Optional dates for validation period
        X_train: Optional training data for distribution comparison
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating regime detector...")
    
    results = {
        'passed': True,
        'issues': [],
        'metrics': {}
    }
    
    # Predict regimes on validation set
    val_regimes = detector.predict(X_val)
    val_confidence = detector.get_regime_confidence(X_val)
    
    # 1. Check regime distribution
    val_distribution = {}
    for regime in range(3):
        pct = (val_regimes == regime).mean() * 100
        val_distribution[REGIME_NAMES[regime]] = pct
        
        if pct < 5:
            results['issues'].append(f"Regime {REGIME_NAMES[regime]} is only {pct:.1f}% in validation")
    
    results['metrics']['validation_distribution'] = val_distribution
    logger.info(f"Validation regime distribution: {val_distribution}")
    
    # 2. Compare to training distribution if provided
    if X_train is not None:
        train_regimes = detector.predict(X_train)
        train_distribution = {}
        for regime in range(3):
            train_distribution[REGIME_NAMES[regime]] = (train_regimes == regime).mean() * 100
        
        results['metrics']['training_distribution'] = train_distribution
        
        # Check for significant distribution shift
        for regime in range(3):
            regime_name = REGIME_NAMES[regime]
            train_pct = train_distribution[regime_name]
            val_pct = val_distribution[regime_name]
            diff = abs(train_pct - val_pct)
            
            if diff > 20:
                results['issues'].append(
                    f"Large distribution shift for {regime_name}: "
                    f"train={train_pct:.1f}%, val={val_pct:.1f}%"
                )
    
    # 3. Check transition matrix consistency
    transmat = detector.get_transition_matrix()
    row_sums = transmat.sum(axis=1)
    
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        results['issues'].append(f"Transition matrix rows don't sum to 1: {row_sums}")
        results['passed'] = False
    
    results['metrics']['transition_matrix'] = transmat.tolist()
    
    # 4. Check for regime stability (no rapid oscillations)
    stability_result = test_regime_persistence(val_regimes)
    results['metrics']['stability'] = stability_result
    
    if stability_result['rapid_switch_pct'] > 5:
        results['issues'].append(
            f"High rapid switching rate: {stability_result['rapid_switch_pct']:.1f}%"
        )
    
    # 5. Check confidence levels
    avg_confidence = val_confidence.mean()
    low_conf_pct = (val_confidence < 0.6).mean() * 100
    
    results['metrics']['avg_confidence'] = float(avg_confidence)
    results['metrics']['low_confidence_pct'] = float(low_conf_pct)
    
    if avg_confidence < 0.7:
        results['issues'].append(f"Low average confidence: {avg_confidence:.2f}")
    
    if low_conf_pct > 20:
        results['issues'].append(f"High percentage of low-confidence periods: {low_conf_pct:.1f}%")
    
    # 6. Check regime durations are realistic
    if dates_val is not None:
        stats = get_regime_statistics(val_regimes, dates_val)
        avg_durations = stats['average_durations']
        
        results['metrics']['average_durations'] = avg_durations
        
        # Check Euphoria duration (expected 30-60 days)
        if 'Euphoria' in avg_durations:
            euphoria_dur = avg_durations['Euphoria']
            if euphoria_dur < 10:
                results['issues'].append(f"Euphoria duration too short: {euphoria_dur:.1f} days")
        
        # Check Capitulation duration (expected 10-30 days)
        if 'Capitulation' in avg_durations:
            cap_dur = avg_durations['Capitulation']
            if cap_dur < 3:
                results['issues'].append(f"Capitulation duration too short: {cap_dur:.1f} days")
    
    # Determine overall pass/fail
    if results['issues']:
        results['passed'] = False
        for issue in results['issues']:
            logger.warning(f"Validation issue: {issue}")
    else:
        logger.info("✓ All validation checks passed")
    
    return results


def test_regime_persistence(regime_sequence: np.ndarray) -> Dict[str, Any]:
    """
    Verify regimes persist for realistic durations.
    
    Flags suspicious rapid switching (potential overfitting).
    
    Args:
        regime_sequence: Array of regime labels
        
    Returns:
        Dictionary with persistence metrics
    """
    logger.info("Testing regime persistence...")
    
    durations = calculate_regime_durations(regime_sequence)
    
    # Count very short regimes (< 3 days)
    very_short = [d for _, d in durations if d < 3]
    short = [d for _, d in durations if d < 5]
    
    total_transitions = len(durations)
    
    result = {
        'total_regime_periods': total_transitions,
        'very_short_periods': len(very_short),
        'short_periods': len(short),
        'rapid_switch_pct': (len(very_short) / total_transitions * 100) if total_transitions > 0 else 0,
        'duration_stats': {
            'mean': np.mean([d for _, d in durations]),
            'median': np.median([d for _, d in durations]),
            'min': min(d for _, d in durations) if durations else 0,
            'max': max(d for _, d in durations) if durations else 0
        }
    }
    
    logger.info(f"Regime persistence: {result['total_regime_periods']} periods, "
                f"{result['rapid_switch_pct']:.1f}% rapid switching")
    
    return result


def visualize_regime_validation(
    detector: RegimeDetector,
    X_val: np.ndarray,
    dates_val: pd.DatetimeIndex,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create validation plots showing regime predictions.
    
    Args:
        detector: Fitted RegimeDetector
        X_val: Validation features
        dates_val: Validation dates
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    logger.info("Creating validation visualization...")
    
    # Get predictions
    regimes = detector.predict(X_val)
    posteriors = detector.predict_proba(X_val)
    confidence = detector.get_regime_confidence(X_val)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. Regime sequence plot
    ax1 = axes[0]
    
    # Create colored background for regimes
    for regime in range(3):
        mask = regimes == regime
        color = REGIME_CHARACTERISTICS[regime]['hex_color']
        name = REGIME_NAMES[regime]
        
        # Find contiguous regions
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        
        for start, end in zip(starts, ends):
            ax1.axvspan(dates_val[start], dates_val[min(end, len(dates_val)-1)], 
                       alpha=0.3, color=color, label=name if start == starts[0] else None)
    
    ax1.set_ylabel('Regime')
    ax1.set_title('Regime Detection - Validation Period')
    ax1.legend(loc='upper right')
    ax1.set_yticks([])
    
    # 2. Posterior probabilities
    ax2 = axes[1]
    for regime in range(3):
        color = REGIME_CHARACTERISTICS[regime]['hex_color']
        name = REGIME_NAMES[regime]
        ax2.plot(dates_val, posteriors[:, regime], color=color, label=name, alpha=0.8)
    
    ax2.set_ylabel('Probability')
    ax2.set_title('Posterior Probabilities')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence with low-confidence highlighting
    ax3 = axes[2]
    ax3.plot(dates_val, confidence, color='blue', alpha=0.7, label='Confidence')
    ax3.axhline(y=0.6, color='red', linestyle='--', label='Threshold (0.6)')
    
    # Highlight low confidence periods
    low_conf = confidence < 0.6
    ax3.fill_between(dates_val, 0, 1, where=low_conf, alpha=0.2, color='red', 
                     label='Low Confidence')
    
    ax3.set_ylabel('Confidence')
    ax3.set_title('Regime Confidence')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Validation plot saved to {save_path}")
    
    return fig


def validate_against_market_events(
    regimes: np.ndarray,
    dates: pd.DatetimeIndex,
    known_events: Optional[Dict[str, Tuple[str, str]]] = None
) -> Dict[str, Any]:
    """
    Validate regime detection against known market events.
    
    Args:
        regimes: Regime sequence
        dates: Date index
        known_events: Dict of event_name -> (start_date, expected_regime)
        
    Returns:
        Dictionary with event validation results
    """
    if known_events is None:
        # Default known events with PRECISE dates
        known_events = {
            'COVID_Crash_Bottom': ('2020-03-23', 'Capitulation'),  # Exact S&P bottom
            'COVID_Recovery_Start': ('2020-04-15', 'Euphoria'),    # Recovery momentum
            '2021_Bull_Peak': ('2021-11-08', 'Euphoria'),          # All-time highs
            '2022_Bear_Start': ('2022-06-13', 'Capitulation'),     # Major breakdown
        }
    
    results = {}
    
    for event_name, (date_str, expected_regime) in known_events.items():
        try:
            event_date = pd.Timestamp(date_str)
            
            # Find closest date in index
            if event_date < dates[0] or event_date > dates[-1]:
                results[event_name] = {
                    'status': 'skipped',
                    'reason': 'Date outside range'
                }
                continue
            
            # Find regime at event date
            idx = dates.get_indexer([event_date], method='nearest')[0]
            actual_regime = REGIME_NAMES.get(regimes[idx], 'Unknown')
            
            match = actual_regime == expected_regime
            
            results[event_name] = {
                'date': date_str,
                'expected': expected_regime,
                'actual': actual_regime,
                'match': match,
                'status': 'pass' if match else 'fail'
            }
            
            if match:
                logger.info(f"✓ {event_name}: correctly detected as {actual_regime}")
            else:
                logger.warning(f"✗ {event_name}: expected {expected_regime}, got {actual_regime}")
                
        except Exception as e:
            results[event_name] = {
                'status': 'error',
                'reason': str(e)
            }
    
    return results


def run_full_validation(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete validation pipeline.
    
    Args:
        model_path: Path to saved model (uses default if None)
        features_path: Path to features file (uses default if None)
        
    Returns:
        Comprehensive validation results
    """
    logger.info("="*60)
    logger.info("RUNNING FULL HMM VALIDATION")
    logger.info("="*60)
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, 'regime_detector.pkl')
    
    detector = RegimeDetector.load_model(model_path)
    
    # Load features
    if features_path is None:
        features_path = config.FEATURES_FILE
    
    features = pd.read_pickle(features_path)
    
    # Use feature columns from detector (saved during training)
    hmm_cols = detector.feature_columns if detector.feature_columns else ['log_returns', 'parkinson_vol']
    logger.info(f"Using HMM feature columns: {hmm_cols}")
    
    # Extract validation period
    val_features = features.loc[config.VAL_START:config.VAL_END]
    train_features = features.loc[config.TRAIN_START:config.TRAIN_END]
    
    X_val_raw = val_features[hmm_cols].dropna()
    X_train_raw = train_features[hmm_cols].dropna()
    
    # CRITICAL: Apply scaler if detector has one (for standardized features)
    if detector.scaler is not None:
        logger.info("Applying scaler to validation data...")
        X_val_scaled = detector.scaler.transform(X_val_raw.values)
        X_train_scaled = detector.scaler.transform(X_train_raw.values)
    else:
        X_val_scaled = X_val_raw.values
        X_train_scaled = X_train_raw.values
    
    logger.info(f"Validation period: {config.VAL_START} to {config.VAL_END}")
    logger.info(f"Validation samples: {len(X_val_raw)}")
    
    # Run validation
    validation_result = validate_regime_detector(
        detector,
        X_val_scaled,
        dates_val=X_val_raw.index,
        X_train=X_train_scaled
    )
    
    # Test persistence
    val_regimes = detector.predict(X_val_scaled)
    persistence_result = test_regime_persistence(val_regimes)
    
    # Validate against known events
    full_features_raw = features[hmm_cols].dropna()
    if detector.scaler is not None:
        full_features_scaled = detector.scaler.transform(full_features_raw.values)
    else:
        full_features_scaled = full_features_raw.values
    full_regimes = detector.predict(full_features_scaled)
    event_validation = validate_against_market_events(
        full_regimes,
        full_features_raw.index
    )
    
    # Create visualization
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    viz_path = os.path.join(config.FIGURES_DIR, 'regime_validation.png')
    visualize_regime_validation(detector, X_val_scaled, X_val_raw.index, save_path=viz_path)
    
    # Compile results
    results = {
        'validation': validation_result,
        'persistence': persistence_result,
        'event_validation': event_validation,
        'visualization_path': viz_path,
        'overall_passed': validation_result['passed'] and persistence_result['rapid_switch_pct'] <= 5
    }
    
    logger.info("="*60)
    if results['overall_passed']:
        logger.info("✓ VALIDATION PASSED")
    else:
        logger.warning("⚠ VALIDATION COMPLETED WITH ISSUES")
    logger.info("="*60)
    
    return results


# Example usage
if __name__ == '__main__':
    try:
        results = run_full_validation()
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall passed: {results['overall_passed']}")
        print(f"\nIssues found: {len(results['validation']['issues'])}")
        for issue in results['validation']['issues']:
            print(f"  - {issue}")
        print(f"\nEvent validation:")
        for event, result in results['event_validation'].items():
            status = result.get('status', 'unknown')
            print(f"  - {event}: {status}")
        
    except FileNotFoundError as e:
        print(f"Required files not found: {e}")
        print("Please run train_hmm.py first to generate model and features.")
    except Exception as e:
        print(f"Validation failed: {e}")
        raise
