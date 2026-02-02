"""
On-Demand Backtest Runner for CHRONOS Dashboard

Provides functionality to run backtests on-demand from the Streamlit dashboard,
with progress indicators and result formatting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    OUTPUTS_DIR, MODELS_DIR, PROCESSED_DATA_DIR,
    INITIAL_CAPITAL, TRANSACTION_COST_RATE, SLIPPAGE_RATE,
    TEST_START, TEST_END
)


def validate_date_range(start_date: str, end_date: str, available_data: pd.DataFrame = None) -> Tuple[bool, str]:
    """
    Validate that the date range is valid for backtesting.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        available_data: Optional DataFrame to check data availability
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        if start >= end:
            return False, "Start date must be before end date"
        
        if (end - start).days < 30:
            return False, "Date range must be at least 30 days"
        
        if available_data is not None and len(available_data) > 0:
            data_start = available_data.index.min()
            data_end = available_data.index.max()
            
            if start < data_start:
                return False, f"Start date {start_date} is before available data ({data_start.date()})"
            
            if end > data_end:
                return False, f"End date {end_date} is after available data ({data_end.date()})"
        
        return True, "Valid date range"
        
    except Exception as e:
        return False, f"Invalid date format: {e}"


def run_backtest_on_demand(
    start_date: str = None,
    end_date: str = None,
    rebalance_freq: str = 'weekly',
    initial_capital: float = None,
    transaction_cost: float = None
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Run a backtest on-demand with specified parameters.
    
    Args:
        start_date: Backtest start date (defaults to TEST_START)
        end_date: Backtest end date (defaults to TEST_END)
        rebalance_freq: Rebalancing frequency ('weekly', 'bi-weekly', 'monthly')
        initial_capital: Initial portfolio value
        transaction_cost: Transaction cost rate
        
    Returns:
        Tuple of (results_df, summary_stats)
    """
    if start_date is None:
        start_date = TEST_START
    if end_date is None:
        end_date = TEST_END
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL
    if transaction_cost is None:
        transaction_cost = TRANSACTION_COST_RATE
    
    try:
        with st.spinner('🔄 Loading models and data...'):
            # Import required modules
            from src.backtest.walk_forward import WalkForwardBacktest
            from src.utils.streamlit_helpers import load_models, load_processed_data
            
            # Load models
            regime_detector, ensemble = load_models()
            
            if regime_detector is None:
                st.error("❌ Regime detector not found. Please train models first.")
                return None, None
            
            if len(ensemble) == 0:
                st.error("❌ Ensemble models not found. Please train models first.")
                return None, None
        
        with st.spinner('📊 Running backtest simulation...'):
            # Initialize backtest engine
            backtester = WalkForwardBacktest(
                initial_capital=initial_capital,
                rebalance_frequency=rebalance_freq,
                transaction_cost=transaction_cost,
                slippage=SLIPPAGE_RATE
            )
            
            # Run backtest
            results_df, summary_stats = backtester.run(
                start_date=start_date,
                end_date=end_date
            )
        
        with st.spinner('💾 Saving results...'):
            # Export results
            from src.backtest.results_exporter import BacktestResultsExporter
            
            exporter = BacktestResultsExporter(
                results_df=results_df,
                summary_stats=summary_stats,
                output_dir=os.path.join(OUTPUTS_DIR, 'backtest')
            )
            exporter.export_all()
        
        st.success('✅ Backtest completed successfully!')
        return results_df, summary_stats
        
    except ImportError as e:
        st.error(f"❌ Required module not found: {e}")
        st.info("Please ensure all dependencies are installed with: pip install -r requirements.txt")
        return None, None
        
    except Exception as e:
        st.error(f"❌ Backtest failed: {e}")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())
        return None, None


def get_rebalance_frequency_options() -> Dict[str, str]:
    """Get available rebalancing frequency options."""
    return {
        'weekly': 'Weekly (Every 5 trading days)',
        'bi-weekly': 'Bi-Weekly (Every 10 trading days)',
        'monthly': 'Monthly (Every 21 trading days)'
    }


def estimate_backtest_time(start_date: str, end_date: str, rebalance_freq: str) -> str:
    """
    Estimate the time to run a backtest based on parameters.
    
    Args:
        start_date: Start date
        end_date: End date
        rebalance_freq: Rebalancing frequency
        
    Returns:
        Estimated time string
    """
    try:
        days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        trading_days = int(days * 252 / 365)  # Approximate
        
        # Rough estimates based on rebalance frequency
        freq_multipliers = {
            'weekly': 1.0,
            'bi-weekly': 0.7,
            'monthly': 0.5
        }
        
        base_time = trading_days * 0.05  # ~0.05 seconds per trading day
        multiplier = freq_multipliers.get(rebalance_freq, 1.0)
        estimated_seconds = base_time * multiplier
        
        if estimated_seconds < 60:
            return f"~{int(estimated_seconds)} seconds"
        else:
            return f"~{int(estimated_seconds / 60)} minutes"
            
    except Exception:
        return "Unknown"


def export_results_for_download(results_df: pd.DataFrame) -> bytes:
    """
    Convert results DataFrame to CSV bytes for download button.
    
    Args:
        results_df: Backtest results DataFrame
        
    Returns:
        CSV data as bytes
    """
    # Reset index to include date as column
    df = results_df.reset_index() if isinstance(results_df.index, pd.DatetimeIndex) else results_df
    return df.to_csv(index=False).encode('utf-8')


def format_backtest_summary(summary_stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Format summary statistics as a display-ready DataFrame.
    
    Args:
        summary_stats: Summary statistics dictionary
        
    Returns:
        Formatted DataFrame for display
    """
    display_metrics = [
        ('Total Return', 'total_return', lambda x: f"{x:.2%}"),
        ('Annualized Return', 'annualized_return', lambda x: f"{x:.2%}"),
        ('Sharpe Ratio', 'sharpe_ratio', lambda x: f"{x:.2f}"),
        ('Sortino Ratio', 'sortino_ratio', lambda x: f"{x:.2f}"),
        ('Max Drawdown', 'max_drawdown', lambda x: f"{x:.2%}"),
        ('Win Rate', 'win_rate', lambda x: f"{x:.2%}"),
        ('Volatility', 'volatility', lambda x: f"{x:.2%}"),
        ('Calmar Ratio', 'calmar_ratio', lambda x: f"{x:.2f}"),
    ]
    
    rows = []
    for display_name, key, formatter in display_metrics:
        value = summary_stats.get(key)
        if value is not None:
            try:
                formatted = formatter(value)
            except:
                formatted = str(value)
            rows.append({'Metric': display_name, 'Value': formatted})
    
    return pd.DataFrame(rows)
