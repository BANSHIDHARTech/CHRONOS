"""
Regime Utilities Module

Provides labeling, duration analysis, interpretation, and export utilities
for market regime detection.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# REGIME CONSTANTS
# ============================================================================

REGIME_NAMES = {
    0: 'Euphoria',
    1: 'Complacency',
    2: 'Capitulation'
}

REGIME_CHARACTERISTICS = {
    0: {
        'name': 'Euphoria',
        'description': 'High returns, low volatility, bullish sentiment',
        'color': 'green',
        'hex_color': '#00C853',
        'expected_returns': 'positive',
        'expected_volatility': 'low'
    },
    1: {
        'name': 'Complacency',
        'description': 'Moderate returns, low volatility, sideways trend',
        'color': 'yellow',
        'hex_color': '#FFD600',
        'expected_returns': 'neutral',
        'expected_volatility': 'low'
    },
    2: {
        'name': 'Capitulation',
        'description': 'Negative returns, high volatility, bearish panic',
        'color': 'red',
        'hex_color': '#D50000',
        'expected_returns': 'negative',
        'expected_volatility': 'high'
    }
}


# ============================================================================
# REGIME DURATION ANALYSIS
# ============================================================================

def calculate_regime_durations(
    regime_sequence: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Calculate duration of consecutive regime periods using run-length encoding.
    
    Args:
        regime_sequence: 1D array of regime labels (0, 1, 2)
        
    Returns:
        List of tuples: [(regime, duration_days), ...]
    """
    if len(regime_sequence) == 0:
        return []
    
    durations = []
    current_regime = regime_sequence[0]
    current_duration = 1
    
    for i in range(1, len(regime_sequence)):
        if regime_sequence[i] == current_regime:
            current_duration += 1
        else:
            durations.append((int(current_regime), current_duration))
            current_regime = regime_sequence[i]
            current_duration = 1
    
    # Add final regime period
    durations.append((int(current_regime), current_duration))
    
    return durations


def get_average_durations(regime_sequence: np.ndarray) -> Dict[int, float]:
    """
    Calculate average duration for each regime.
    
    Args:
        regime_sequence: 1D array of regime labels
        
    Returns:
        Dictionary: {regime_id: average_duration_days}
    """
    durations = calculate_regime_durations(regime_sequence)
    
    # Group by regime
    regime_durations: Dict[int, List[int]] = {}
    for regime, duration in durations:
        if regime not in regime_durations:
            regime_durations[regime] = []
        regime_durations[regime].append(duration)
    
    # Calculate averages
    averages = {
        regime: np.mean(durs)
        for regime, durs in regime_durations.items()
    }
    
    return averages


def get_regime_statistics(
    regime_sequence: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive regime statistics.
    
    Args:
        regime_sequence: 1D array of regime labels
        dates: Optional DatetimeIndex for date-based analysis
        
    Returns:
        Dictionary with comprehensive statistics
    """
    regime_sequence = np.asarray(regime_sequence)
    durations = calculate_regime_durations(regime_sequence)
    
    statistics: Dict[str, Any] = {
        'total_observations': len(regime_sequence),
        'unique_regimes': len(np.unique(regime_sequence)),
        'regime_counts': {},
        'regime_frequency': {},
        'average_durations': {},
        'min_durations': {},
        'max_durations': {},
        'num_regime_periods': {}
    }
    
    # Calculate counts and frequencies
    for regime in range(3):
        count = int((regime_sequence == regime).sum())
        statistics['regime_counts'][REGIME_NAMES[regime]] = count
        statistics['regime_frequency'][REGIME_NAMES[regime]] = count / len(regime_sequence) if len(regime_sequence) > 0 else 0
    
    # Calculate duration statistics per regime
    for regime in range(3):
        regime_durs = [d for r, d in durations if r == regime]
        regime_name = REGIME_NAMES[regime]
        
        if regime_durs:
            statistics['average_durations'][regime_name] = float(np.mean(regime_durs))
            statistics['min_durations'][regime_name] = int(min(regime_durs))
            statistics['max_durations'][regime_name] = int(max(regime_durs))
            statistics['num_regime_periods'][regime_name] = len(regime_durs)
        else:
            statistics['average_durations'][regime_name] = 0.0
            statistics['min_durations'][regime_name] = 0
            statistics['max_durations'][regime_name] = 0
            statistics['num_regime_periods'][regime_name] = 0
    
    # Add longest/shortest periods with dates if available
    if dates is not None and len(dates) == len(regime_sequence):
        statistics['longest_periods'] = {}
        statistics['shortest_periods'] = {}
        
        # Track regime periods with start/end dates
        idx = 0
        for regime, duration in durations:
            regime_name = REGIME_NAMES[regime]
            start_date = dates[idx]
            end_date = dates[min(idx + duration - 1, len(dates) - 1)]
            
            period_info = {
                'duration': duration,
                'start_date': str(start_date.date()),
                'end_date': str(end_date.date())
            }
            
            # Update longest
            if regime_name not in statistics['longest_periods']:
                statistics['longest_periods'][regime_name] = period_info
            elif duration > statistics['longest_periods'][regime_name]['duration']:
                statistics['longest_periods'][regime_name] = period_info
            
            # Update shortest
            if regime_name not in statistics['shortest_periods']:
                statistics['shortest_periods'][regime_name] = period_info
            elif duration < statistics['shortest_periods'][regime_name]['duration']:
                statistics['shortest_periods'][regime_name] = period_info
            
            idx += duration
    
    return statistics


def identify_regime_switches(
    regime_sequence: np.ndarray,
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Detect all regime transitions with timestamps.
    
    Args:
        regime_sequence: 1D array of regime labels
        dates: DatetimeIndex aligned with regime_sequence
        
    Returns:
        DataFrame with columns: ['date', 'from_regime', 'to_regime', 'transition_type']
    """
    regime_sequence = np.asarray(regime_sequence)
    
    if len(regime_sequence) != len(dates):
        raise ValueError("regime_sequence and dates must have same length")
    
    switches = []
    
    for i in range(1, len(regime_sequence)):
        if regime_sequence[i] != regime_sequence[i-1]:
            from_regime = int(regime_sequence[i-1])
            to_regime = int(regime_sequence[i])
            
            # Determine transition type
            transition_type = _get_transition_type(from_regime, to_regime)
            
            switches.append({
                'date': dates[i],
                'from_regime': REGIME_NAMES[from_regime],
                'to_regime': REGIME_NAMES[to_regime],
                'from_regime_id': from_regime,
                'to_regime_id': to_regime,
                'transition_type': transition_type
            })
    
    return pd.DataFrame(switches)


def _get_transition_type(from_regime: int, to_regime: int) -> str:
    """
    Label transition type based on from/to regimes.
    
    Regime 0 = Euphoria (bull)
    Regime 1 = Complacency (neutral)
    Regime 2 = Capitulation (bear)
    """
    transitions = {
        (0, 1): 'bull_to_neutral',
        (0, 2): 'bull_to_bear',
        (1, 0): 'neutral_to_bull',
        (1, 2): 'neutral_to_bear',
        (2, 0): 'bear_to_bull',
        (2, 1): 'bear_to_neutral'
    }
    
    return transitions.get((from_regime, to_regime), 'unknown')


# ============================================================================
# REGIME LABELING AND INTERPRETATION
# ============================================================================

def label_regimes(regime_array: np.ndarray) -> np.ndarray:
    """
    Convert numeric labels to string names.
    
    Args:
        regime_array: Array of regime IDs (0, 1, 2)
        
    Returns:
        Array of regime names
    """
    return np.array([REGIME_NAMES.get(int(r), 'Unknown') for r in regime_array])


def get_regime_color(regime_id: int) -> str:
    """
    Return matplotlib-compatible color for visualization.
    
    Args:
        regime_id: Regime identifier (0, 1, 2)
        
    Returns:
        Color string
    """
    return REGIME_CHARACTERISTICS.get(regime_id, {}).get('color', 'gray')


def get_regime_hex_color(regime_id: int) -> str:
    """
    Return hex color code for visualization.
    
    Args:
        regime_id: Regime identifier (0, 1, 2)
        
    Returns:
        Hex color string
    """
    return REGIME_CHARACTERISTICS.get(regime_id, {}).get('hex_color', '#808080')


def interpret_current_regime(
    regime_id: int,
    confidence: float,
    transition_probs: Optional[np.ndarray] = None
) -> str:
    """
    Generate natural language description of current regime.
    
    Args:
        regime_id: Current regime (0, 1, 2)
        confidence: Posterior probability confidence (0-1)
        transition_probs: Optional row of transition matrix for this regime
        
    Returns:
        Formatted description string
    """
    regime_name = REGIME_NAMES.get(regime_id, 'Unknown')
    regime_info = REGIME_CHARACTERISTICS.get(regime_id, {})
    description = regime_info.get('description', 'Unknown regime characteristics')
    
    interpretation = (
        f"Market is in {regime_name} regime ({confidence:.0%} confidence).\n"
        f"Characteristics: {description}"
    )
    
    if transition_probs is not None and len(transition_probs) == 3:
        # Find most likely next regime (excluding staying in same)
        other_probs = []
        for i, prob in enumerate(transition_probs):
            if i != regime_id:
                other_probs.append((i, prob))
        
        if other_probs:
            most_likely = max(other_probs, key=lambda x: x[1])
            next_regime_name = REGIME_NAMES.get(most_likely[0], 'Unknown')
            next_prob = most_likely[1]
            
            interpretation += (
                f"\n{next_prob:.0%} probability of transitioning to {next_regime_name} next period."
            )
            
            # Self-transition (persistence)
            persistence = transition_probs[regime_id]
            interpretation += f"\n{persistence:.0%} probability of staying in {regime_name}."
    
    return interpretation


def get_regime_description(regime_id: int) -> Dict[str, Any]:
    """
    Get full characteristics of a regime.
    
    Args:
        regime_id: Regime identifier
        
    Returns:
        Dictionary with regime characteristics
    """
    return REGIME_CHARACTERISTICS.get(regime_id, {
        'name': 'Unknown',
        'description': 'Unknown regime',
        'color': 'gray',
        'hex_color': '#808080',
        'expected_returns': 'unknown',
        'expected_volatility': 'unknown'
    })


# ============================================================================
# DATA FILTERING AND ANALYSIS
# ============================================================================

def filter_by_regime(
    data: pd.DataFrame,
    regime_labels: np.ndarray,
    target_regime: Union[int, str]
) -> pd.DataFrame:
    """
    Filter DataFrame to specific regime periods.
    
    Args:
        data: DataFrame with DatetimeIndex
        regime_labels: Array of regime labels aligned with data index
        target_regime: Regime ID (0,1,2) or name ('Euphoria', etc.)
        
    Returns:
        Filtered DataFrame
    """
    if isinstance(target_regime, str):
        # Convert name to ID
        target_id = None
        for rid, name in REGIME_NAMES.items():
            if name.lower() == target_regime.lower():
                target_id = rid
                break
        if target_id is None:
            raise ValueError(f"Unknown regime name: {target_regime}")
    else:
        target_id = target_regime
    
    regime_labels = np.asarray(regime_labels)
    
    if len(regime_labels) != len(data):
        raise ValueError("regime_labels must have same length as data")
    
    mask = regime_labels == target_id
    filtered = data[mask].copy()
    
    logger.info(
        f"Filtered to regime {REGIME_NAMES.get(target_id, target_id)}: "
        f"{mask.sum()} observations ({mask.mean()*100:.1f}%)"
    )
    
    return filtered


def get_regime_summary_statistics(
    data: pd.DataFrame,
    regime_labels: np.ndarray,
    returns_column: str = 'log_returns'
) -> pd.DataFrame:
    """
    Calculate summary statistics per regime.
    
    Args:
        data: DataFrame with feature columns
        regime_labels: Array of regime labels
        returns_column: Column name for returns
        
    Returns:
        Summary DataFrame with statistics per regime
    """
    regime_labels = np.asarray(regime_labels)
    
    if returns_column not in data.columns:
        logger.warning(f"Returns column '{returns_column}' not found. Using first column.")
        returns_column = data.columns[0]
    
    summaries = []
    
    for regime in range(3):
        mask = regime_labels == regime
        regime_data = data[mask]
        
        if len(regime_data) == 0:
            continue
        
        returns = regime_data[returns_column].dropna()
        
        # Calculate statistics
        mean_return = returns.mean() * 252  # Annualized
        std_return = returns.std() * np.sqrt(252)  # Annualized
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        summaries.append({
            'regime': REGIME_NAMES[regime],
            'regime_id': regime,
            'observations': len(regime_data),
            'pct_of_total': len(regime_data) / len(data) * 100,
            'mean_return_ann': mean_return,
            'std_return_ann': std_return,
            'sharpe_ratio': sharpe,
            'min_return': returns.min(),
            'max_return': returns.max(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis()
        })
    
    df = pd.DataFrame(summaries)
    
    # Validate financial interpretation
    if len(df) == 3:
        euphoria_sharpe = df[df['regime'] == 'Euphoria']['sharpe_ratio'].values[0]
        capitulation_std = df[df['regime'] == 'Capitulation']['std_return_ann'].values[0]
        euphoria_std = df[df['regime'] == 'Euphoria']['std_return_ann'].values[0]
        
        if euphoria_sharpe < df['sharpe_ratio'].max():
            logger.warning("Euphoria regime does not have highest Sharpe ratio - check regime ordering")
        
        if capitulation_std < euphoria_std:
            logger.warning("Capitulation has lower volatility than Euphoria - check regime ordering")
    
    return df


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_regime_analysis(
    regime_detector,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: str
) -> Dict[str, str]:
    """
    Generate comprehensive regime analysis report.
    
    Args:
        regime_detector: Fitted RegimeDetector instance
        X: Feature matrix
        dates: DatetimeIndex aligned with X
        output_dir: Output directory path
        
    Returns:
        Dictionary with paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Get predictions
    regimes = regime_detector.predict(X)
    
    # 1. Save transition matrix
    transmat_df = regime_detector.calculate_transition_probabilities()
    transmat_path = output_dir / 'transition_matrix.csv'
    transmat_df.to_csv(transmat_path)
    saved_files['transition_matrix'] = str(transmat_path)
    logger.info(f"Saved transition matrix to {transmat_path}")
    
    # 2. Save regime durations
    durations = calculate_regime_durations(regimes)
    durations_df = pd.DataFrame(durations, columns=['regime_id', 'duration_days'])
    durations_df['regime_name'] = durations_df['regime_id'].map(REGIME_NAMES)
    durations_path = output_dir / 'regime_durations.csv'
    durations_df.to_csv(durations_path, index=False)
    saved_files['regime_durations'] = str(durations_path)
    logger.info(f"Saved regime durations to {durations_path}")
    
    # 3. Save regime statistics
    statistics = get_regime_statistics(regimes, dates)
    stats_path = output_dir / 'regime_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2, default=str)
    saved_files['regime_statistics'] = str(stats_path)
    logger.info(f"Saved regime statistics to {stats_path}")
    
    # 4. Save regime timeline
    timeline_df = pd.DataFrame({
        'date': dates,
        'regime_id': regimes,
        'regime_name': label_regimes(regimes)
    })
    
    # Add confidence scores
    confidence = regime_detector.get_regime_confidence(X)
    timeline_df['confidence'] = confidence
    
    timeline_path = output_dir / 'regime_timeline.csv'
    timeline_df.to_csv(timeline_path, index=False)
    saved_files['regime_timeline'] = str(timeline_path)
    logger.info(f"Saved regime timeline to {timeline_path}")
    
    # 5. Save regime switches
    switches_df = identify_regime_switches(regimes, dates)
    if len(switches_df) > 0:
        switches_path = output_dir / 'regime_switches.csv'
        switches_df.to_csv(switches_path, index=False)
        saved_files['regime_switches'] = str(switches_path)
        logger.info(f"Saved regime switches to {switches_path}")
    
    # 6. Save model info
    model_info = regime_detector.get_model_info()
    model_info_path = output_dir / 'model_info.json'
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    saved_files['model_info'] = str(model_info_path)
    logger.info(f"Saved model info to {model_info_path}")
    
    logger.info(f"Regime analysis exported to {output_dir}")
    
    return saved_files


# Example usage
if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Simulate regime sequence
    regime_sequence = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')
    
    # Test duration calculations
    durations = calculate_regime_durations(regime_sequence)
    print(f"Number of regime periods: {len(durations)}")
    
    avg_durations = get_average_durations(regime_sequence)
    print(f"\nAverage durations: {avg_durations}")
    
    # Test statistics
    stats = get_regime_statistics(regime_sequence, dates)
    print(f"\nRegime frequency: {stats['regime_frequency']}")
    
    # Test regime switches
    switches = identify_regime_switches(regime_sequence, dates)
    print(f"\nNumber of regime switches: {len(switches)}")
    
    # Test interpretation
    interpretation = interpret_current_regime(0, 0.85, np.array([0.9, 0.07, 0.03]))
    print(f"\n{interpretation}")
