"""
Central Configuration for CHRONOS Quantitative Finance System

This module serves as the single source of truth for all configuration parameters.
It enforces strict temporal boundaries and anti-leakage settings to ensure
institutional-grade data integrity.
"""

from dataclasses import dataclass
from typing import Dict, List
import os
from datetime import datetime


# ============================================================================
# DATA SOURCES CONFIGURATION
# ============================================================================

TICKERS = {
    '^GSPC': 'S&P 500 Index',
    '^VIX': 'CBOE Volatility Index',
    '^TNX': '10-Year Treasury Yield',
    'GC=F': 'Gold Futures',
    'SPY': 'S&P 500 ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'GLD': 'SPDR Gold Trust ETF'
}

# FRED Economic Indicators (optional for Phase 1)
FRED_SERIES = {
    'T10Y2Y': 'Treasury 10Y-2Y Spread',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS2': '2-Year Treasury Constant Maturity Rate'
}

# Data download date ranges
DATA_START_DATE = '2019-01-01'
DATA_END_DATE = '2024-12-31'


# ============================================================================
# TIME SPLIT CONFIGURATION
# ============================================================================

# Training period
TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31'

# Validation period
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'

# Test period
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'


# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Technical indicators
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Risk metrics
ROLLING_SHARPE_WINDOW = 63  # 3 months trading days (~252/4)
ROLLING_MAX_DD_WINDOW = 126  # 6 months trading days (~252/2)

# Volatility measures
VIX_PERCENTILE_WINDOW = 20
PARKINSON_VOL_WINDOW = 20

# Bollinger Bands
BOLLINGER_BAND_WINDOW = 20
BOLLINGER_BAND_STD = 2

# Average True Range
ATR_PERIOD = 14

# Moving averages
SMA_WINDOWS = [20, 50]
EMA_WINDOWS = [12, 26]

# Risk-free rate (annualized)
RISK_FREE_RATE = 0.02


# ============================================================================
# TARGET ENGINEERING PARAMETERS
# ============================================================================

# Forward return horizon
FORWARD_RETURN_DAYS = 5

# Quintile classification labels (1=bottom 20%, 5=top 20%)
QUINTILE_LABELS = [1, 2, 3, 4, 5]


# ============================================================================
# FILE PATHS
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')

# Processed data files
FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'features.pkl')
TARGETS_FILE = os.path.join(PROCESSED_DATA_DIR, 'targets.pkl')
ALIGNED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'aligned_data.pkl')
TRAIN_SPLIT_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_split.pkl')
VAL_SPLIT_FILE = os.path.join(PROCESSED_DATA_DIR, 'val_split.pkl')
TEST_SPLIT_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_split.pkl')


# ============================================================================
# ANTI-LEAKAGE SETTINGS
# ============================================================================

# Enforce that all features are shifted by at least 1 period
ENFORCE_SHIFT = True

# Require explicit min_periods in all rolling operations
MIN_PERIODS_REQUIRED = True

# Minimum history required (in trading days) for valid feature calculation
MIN_HISTORY_DAYS = 252  # 1 year of trading days


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_date_ranges() -> bool:
    """
    Validate that date ranges are logically consistent.
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If date ranges are inconsistent
    """
    # Parse dates
    data_start = datetime.strptime(DATA_START_DATE, '%Y-%m-%d')
    data_end = datetime.strptime(DATA_END_DATE, '%Y-%m-%d')
    train_start = datetime.strptime(TRAIN_START, '%Y-%m-%d')
    train_end = datetime.strptime(TRAIN_END, '%Y-%m-%d')
    val_start = datetime.strptime(VAL_START, '%Y-%m-%d')
    val_end = datetime.strptime(VAL_END, '%Y-%m-%d')
    test_start = datetime.strptime(TEST_START, '%Y-%m-%d')
    test_end = datetime.strptime(TEST_END, '%Y-%m-%d')
    
    # Check data range encompasses all splits
    if train_start < data_start:
        raise ValueError(f"TRAIN_START ({TRAIN_START}) is before DATA_START_DATE ({DATA_START_DATE})")
    
    if test_end > data_end:
        raise ValueError(f"TEST_END ({TEST_END}) is after DATA_END_DATE ({DATA_END_DATE})")
    
    # Check chronological order of splits
    if train_end >= val_start:
        raise ValueError(f"TRAIN_END ({TRAIN_END}) must be before VAL_START ({VAL_START})")
    
    if val_end >= test_start:
        raise ValueError(f"VAL_END ({VAL_END}) must be before TEST_START ({TEST_START})")
    
    # Check that each split has reasonable duration
    train_days = (train_end - train_start).days
    val_days = (val_end - val_start).days
    test_days = (test_end - test_start).days
    
    if train_days < MIN_HISTORY_DAYS:
        raise ValueError(f"Training period ({train_days} days) is shorter than MIN_HISTORY_DAYS ({MIN_HISTORY_DAYS})")
    
    if val_days < 30:
        raise ValueError(f"Validation period ({val_days} days) is too short (minimum 30 days)")
    
    if test_days < 30:
        raise ValueError(f"Test period ({test_days} days) is too short (minimum 30 days)")
    
    return True


def validate_feature_parameters() -> bool:
    """
    Validate that feature engineering parameters are reasonable.
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Check RSI period
    if not (2 <= RSI_PERIOD <= 50):
        raise ValueError(f"RSI_PERIOD ({RSI_PERIOD}) must be between 2 and 50")
    
    # Check MACD parameters
    if MACD_FAST >= MACD_SLOW:
        raise ValueError(f"MACD_FAST ({MACD_FAST}) must be less than MACD_SLOW ({MACD_SLOW})")
    
    # Check rolling window sizes
    if ROLLING_SHARPE_WINDOW < 20:
        raise ValueError(f"ROLLING_SHARPE_WINDOW ({ROLLING_SHARPE_WINDOW}) should be at least 20 days")
    
    if ROLLING_MAX_DD_WINDOW < 20:
        raise ValueError(f"ROLLING_MAX_DD_WINDOW ({ROLLING_MAX_DD_WINDOW}) should be at least 20 days")
    
    # Check Bollinger Band parameters
    if BOLLINGER_BAND_WINDOW < 10:
        raise ValueError(f"BOLLINGER_BAND_WINDOW ({BOLLINGER_BAND_WINDOW}) should be at least 10 days")
    
    if not (1 <= BOLLINGER_BAND_STD <= 3):
        raise ValueError(f"BOLLINGER_BAND_STD ({BOLLINGER_BAND_STD}) should be between 1 and 3")
    
    # Check moving average windows
    if not all(w > 0 for w in SMA_WINDOWS):
        raise ValueError(f"All SMA_WINDOWS must be positive: {SMA_WINDOWS}")
    
    if not all(w > 0 for w in EMA_WINDOWS):
        raise ValueError(f"All EMA_WINDOWS must be positive: {EMA_WINDOWS}")
    
    return True


def validate_target_parameters() -> bool:
    """
    Validate target engineering parameters.
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Check forward return horizon
    if not (1 <= FORWARD_RETURN_DAYS <= 20):
        raise ValueError(f"FORWARD_RETURN_DAYS ({FORWARD_RETURN_DAYS}) should be between 1 and 20")
    
    # Check quintile labels
    if len(QUINTILE_LABELS) != 5:
        raise ValueError(f"QUINTILE_LABELS must have exactly 5 elements, got {len(QUINTILE_LABELS)}")
    
    if sorted(QUINTILE_LABELS) != QUINTILE_LABELS:
        raise ValueError(f"QUINTILE_LABELS must be in ascending order: {QUINTILE_LABELS}")
    
    return True


def validate_all() -> bool:
    """
    Run all configuration validations.
    
    Returns:
        bool: True if all validations pass
    """
    validate_date_ranges()
    validate_feature_parameters()
    validate_target_parameters()
    return True


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 80)
    print("CHRONOS CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nTickers: {', '.join(TICKERS.keys())}")
    print(f"Data Range: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"\nTrain Split: {TRAIN_START} to {TRAIN_END}")
    print(f"Val Split: {VAL_START} to {VAL_END}")
    print(f"Test Split: {TEST_START} to {TEST_END}")
    print(f"\nForward Return Horizon: {FORWARD_RETURN_DAYS} days")
    print(f"Anti-Leakage Enforcement: {'ENABLED' if ENFORCE_SHIFT else 'DISABLED'}")
    print(f"Min Periods Required: {'YES' if MIN_PERIODS_REQUIRED else 'NO'}")
    print("=" * 80)


# Run validation on import
if __name__ == '__main__':
    try:
        validate_all()
        print("[OK] Configuration validation passed")
        print_config_summary()
    except ValueError as e:
        print(f"[FAIL] Configuration validation failed: {e}")


# ============================================================================
# XGBOOST CONFIGURATION
# ============================================================================

XGBOOST_CONFIG = {
    'max_depth': 3,
    'reg_lambda': 1,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'early_stopping_rounds': 10,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
}

MODEL_PATHS = {
    'ensemble_dir': os.path.join(BASE_DIR, 'models'),
    'regime_0': os.path.join(BASE_DIR, 'models', 'xgb_regime_0_euphoria.json'),
    'regime_1': os.path.join(BASE_DIR, 'models', 'xgb_regime_1_complacency.json'),
    'regime_2': os.path.join(BASE_DIR, 'models', 'xgb_regime_2_capitulation.json'),
    'metadata': os.path.join(BASE_DIR, 'models', 'ensemble_metadata.json'),
}

EVALUATION_CONFIG = {
    'confidence_bins': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'min_samples_per_regime': 100,
}


# ============================================================================
# PORTFOLIO OPTIMIZATION CONFIGURATION
# ============================================================================

# Portfolio assets
PORTFOLIO_ASSETS = ['SPY', 'TLT', 'GLD']

# Initial capital for backtesting
INITIAL_CAPITAL = 100000  # $100,000

# Rebalancing frequency: 'daily', 'weekly', 'bi-weekly', 'monthly'
REBALANCE_FREQUENCY = 'weekly'

# Transaction costs (as fractions)
TRANSACTION_COST_RATE = 0.001  # 10 basis points per trade
SLIPPAGE_RATE = 0.0005  # 5 basis points slippage

# Regime-specific portfolio constraints
# Each regime defines (min_weight, max_weight) bounds per asset
REGIME_CONSTRAINTS = {
    0: {  # Euphoria - Risk-on, higher equity allocation
        'SPY': (0.60, 0.80),
        'TLT': (0.10, 0.30),
        'GLD': (0.10, 0.30)
    },
    1: {  # Complacency - Balanced allocation
        'SPY': (0.40, 0.60),
        'TLT': (0.20, 0.40),
        'GLD': (0.10, 0.30)
    },
    2: {  # Capitulation - Risk-off, defensive allocation
        'SPY': (0.00, 0.20),
        'TLT': (0.50, 0.70),
        'GLD': (0.20, 0.40)
    }
}

# Confidence-based weight adjustment thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for weight adjustment
BULLISH_QUINTILE = 5  # Top quintile (most bullish prediction)
BEARISH_QUINTILE = 1  # Bottom quintile (most bearish prediction)
EQUITY_BOOST_FACTOR = 1.2  # 20% increase in SPY weight when bullish
EQUITY_CUT_FACTOR = 0.5    # 50% decrease in SPY weight when bearish

