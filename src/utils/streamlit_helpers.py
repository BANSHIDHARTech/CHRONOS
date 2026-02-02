"""
Streamlit Helper Functions for CHRONOS Dashboard

Provides cached data loading, model loading, and utility functions
for the Streamlit dashboard application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import os
import sys
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    OUTPUTS_DIR, MODELS_DIR, PROCESSED_DATA_DIR,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END
)


# ============================================================================
# BACKTEST RESULTS LOADING
# ============================================================================

@st.cache_data(ttl=3600)
def load_backtest_results(results_dir: str = None) -> Dict[str, Any]:
    """
    Load all backtest results with Streamlit caching.
    
    Args:
        results_dir: Directory containing backtest results (defaults to outputs/backtest)
        
    Returns:
        Dictionary containing:
        - dashboard_data: Main dashboard JSON data
        - summary_statistics: Summary stats dictionary
        - results_df: Full results DataFrame
        - regime_performance: Performance by regime
    """
    if results_dir is None:
        results_dir = os.path.join(OUTPUTS_DIR, 'backtest')
    
    results = {}
    
    # Load dashboard data JSON
    dashboard_path = os.path.join(results_dir, 'dashboard_data.json')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            results['dashboard_data'] = json.load(f)
    else:
        results['dashboard_data'] = None
    
    # Load summary statistics
    summary_path = os.path.join(results_dir, 'summary_statistics.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary_statistics'] = json.load(f)
    else:
        results['summary_statistics'] = None
    
    # Load results DataFrame
    results_csv_path = os.path.join(results_dir, 'backtest_results.csv')
    if os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        # Normalize weight column names to consistent format (weight_SPY, weight_TLT, weight_GLD)
        df = normalize_weight_columns(df)
        results['results_df'] = df
    else:
        results['results_df'] = None
    
    # Load regime performance
    regime_perf_path = os.path.join(results_dir, 'regime_performance.csv')
    if os.path.exists(regime_perf_path):
        results['regime_performance'] = pd.read_csv(regime_perf_path)
    else:
        results['regime_performance'] = None
    
    # Load trade log
    trade_log_path = os.path.join(results_dir, 'trade_log.csv')
    if os.path.exists(trade_log_path):
        results['trade_log'] = pd.read_csv(trade_log_path)
    else:
        results['trade_log'] = None
    
    # Load crisis analysis
    crisis_path = os.path.join(results_dir, 'crisis_analysis.csv')
    if os.path.exists(crisis_path):
        results['crisis_analysis'] = pd.read_csv(crisis_path)
    else:
        results['crisis_analysis'] = None
    
    return results


def normalize_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weight column names to consistent format (weight_SPY, weight_TLT, weight_GLD).
    
    Handles various naming conventions from different exporters:
    - spy_weight -> weight_SPY
    - SPY_weight -> weight_SPY
    - w_spy -> weight_SPY
    
    Args:
        df: DataFrame with weight columns
        
    Returns:
        DataFrame with normalized column names
    """
    # Mapping of possible source column patterns to target names
    weight_mappings = {
        'weight_SPY': ['spy_weight', 'SPY_weight', 'w_spy', 'w_SPY', 'spy_wt', 'SPY'],
        'weight_TLT': ['tlt_weight', 'TLT_weight', 'w_tlt', 'w_TLT', 'tlt_wt', 'TLT'],
        'weight_GLD': ['gld_weight', 'GLD_weight', 'w_gld', 'w_GLD', 'gld_wt', 'GLD'],
    }
    
    rename_map = {}
    for target_col, source_patterns in weight_mappings.items():
        # Skip if target column already exists
        if target_col in df.columns:
            continue
        # Look for source column matching any pattern
        for pattern in source_patterns:
            if pattern in df.columns:
                rename_map[pattern] = target_col
                break
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


@st.cache_resource
def load_models(models_dir: str = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load CHRONOS models with Streamlit resource caching.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        Tuple of (regime_detector, ensemble_models_dict)
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    regime_detector = None
    ensemble = {}
    
    # Load regime detector (HMM)
    detector_path = os.path.join(models_dir, 'regime_detector.pkl')
    if os.path.exists(detector_path):
        try:
            with open(detector_path, 'rb') as f:
                regime_detector = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading regime detector: {e}")
    
    # Load XGBoost ensemble models
    import xgboost as xgb
    
    for regime_id in [0, 1, 2]:
        regime_names = {0: 'euphoria', 1: 'complacency', 2: 'capitulation'}
        model_path = os.path.join(models_dir, f'xgb_regime_{regime_id}_{regime_names[regime_id]}.json')
        
        if os.path.exists(model_path):
            try:
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                ensemble[regime_id] = model
            except Exception as e:
                st.warning(f"Could not load model for regime {regime_id}: {e}")
    
    # Load ensemble metadata
    metadata_path = os.path.join(models_dir, 'ensemble_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            ensemble['metadata'] = json.load(f)
    
    return regime_detector, ensemble


@st.cache_data
def load_processed_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load processed features and returns data.
    
    Returns:
        Tuple of (features_df, returns_df)
    """
    features_df = None
    returns_df = None
    
    # Load features
    features_path = os.path.join(PROCESSED_DATA_DIR, 'features.pkl')
    if os.path.exists(features_path):
        try:
            with open(features_path, 'rb') as f:
                features_df = pickle.load(f)
        except Exception as e:
            # Try CSV fallback
            csv_path = os.path.join(PROCESSED_DATA_DIR, 'features.csv')
            if os.path.exists(csv_path):
                features_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Load returns
    returns_path = os.path.join(PROCESSED_DATA_DIR, 'asset_returns.pkl')
    if os.path.exists(returns_path):
        try:
            with open(returns_path, 'rb') as f:
                returns_df = pickle.load(f)
        except Exception as e:
            pass
    
    return features_df, returns_df


# ============================================================================
# SHAP DATA LOADING
# ============================================================================

@st.cache_data
def load_shap_plots(regime_id: int) -> Dict[str, str]:
    """
    Load SHAP plot paths for a specific regime.
    
    Args:
        regime_id: Regime identifier (0, 1, or 2)
        
    Returns:
        Dictionary mapping plot type to file path
    """
    regime_names = {0: 'euphoria', 1: 'complacency', 2: 'capitulation'}
    regime_name = regime_names.get(regime_id, str(regime_id))
    
    shap_dir = os.path.join(OUTPUTS_DIR, 'shap', f'regime_{regime_id}_{regime_name}')
    
    plots = {}
    plot_types = ['summary', 'beeswarm', 'bar', 'waterfall']
    
    for plot_type in plot_types:
        # Try both naming conventions
        for ext in ['.png', '.jpg']:
            path = os.path.join(shap_dir, f'{plot_type}_plot{ext}')
            if os.path.exists(path):
                plots[plot_type] = path
                break
            # Alternative naming
            alt_path = os.path.join(shap_dir, f'shap_{plot_type}{ext}')
            if os.path.exists(alt_path):
                plots[plot_type] = alt_path
                break
    
    return plots


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_date_range(start_date, end_date, data_df: pd.DataFrame = None) -> bool:
    """
    Validate that date range is valid and within available data.
    
    Args:
        start_date: Start date
        end_date: End date
        data_df: Optional DataFrame to check against
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    if data_df is not None and len(data_df) > 0:
        data_start = data_df.index.min()
        data_end = data_df.index.max()
        
        if pd.Timestamp(start_date) < data_start:
            raise ValueError(f"Start date {start_date} is before available data ({data_start})")
        
        if pd.Timestamp(end_date) > data_end:
            raise ValueError(f"End date {end_date} is after available data ({data_end})")
    
    return True


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_for_download(results_df: pd.DataFrame) -> bytes:
    """
    Convert results DataFrame to CSV bytes for download.
    
    Args:
        results_df: Results DataFrame
        
    Returns:
        CSV data as bytes
    """
    return results_df.to_csv().encode('utf-8')


def export_summary_for_download(summary_stats: Dict[str, Any]) -> bytes:
    """
    Convert summary statistics to JSON bytes for download.
    
    Args:
        summary_stats: Summary statistics dictionary
        
    Returns:
        JSON data as bytes
    """
    return json.dumps(summary_stats, indent=2, default=str).encode('utf-8')


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string."""
    return f"${value:,.{decimals}f}"


def format_ratio(value: float, decimals: int = 2) -> str:
    """Format value as ratio string."""
    return f"{value:.{decimals}f}"


def get_regime_color(regime_id: int) -> str:
    """Get color for regime display."""
    colors = {
        0: '#00C853',  # Euphoria - Green
        1: '#FFD600',  # Complacency - Yellow
        2: '#D50000',  # Capitulation - Red
    }
    return colors.get(regime_id, '#808080')


def get_regime_name(regime_id: int) -> str:
    """Get name for regime."""
    names = {
        0: 'Euphoria',
        1: 'Complacency',
        2: 'Capitulation'
    }
    return names.get(regime_id, 'Unknown')


# ============================================================================
# ERROR HANDLING
# ============================================================================

def show_data_error(component_name: str, error: Exception = None):
    """Display user-friendly error message for data loading failures."""
    st.error(f"❌ Unable to load {component_name}")
    if error:
        with st.expander("Error Details"):
            st.code(str(error))


def show_missing_data_warning(component_name: str, action: str = "run the backtest first"):
    """Display warning for missing data with suggested action."""
    st.warning(f"⚠️ {component_name} not found. Please {action}.")


# ============================================================================
# MODEL INFO DISPLAY
# ============================================================================

def get_model_info(regime_detector, ensemble: Dict) -> Dict[str, Any]:
    """
    Extract model information for display.
    
    Args:
        regime_detector: Loaded regime detector
        ensemble: Loaded ensemble models
        
    Returns:
        Dictionary with model information
    """
    info = {
        'regime_detector_loaded': regime_detector is not None,
        'ensemble_models_loaded': len([k for k in ensemble if isinstance(k, int)]),
        'training_period': f"{TRAIN_START} to {TRAIN_END}",
        'test_period': f"{TEST_START} to {TEST_END}",
    }
    
    if 'metadata' in ensemble:
        meta = ensemble['metadata']
        info['training_date'] = meta.get('training_date', 'Unknown')
        info['feature_count'] = meta.get('n_features', 'Unknown')
    
    return info
