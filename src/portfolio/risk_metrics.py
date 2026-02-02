"""
Comprehensive Risk Metrics Calculator

This module provides institutional-grade risk metrics including Sharpe, Sortino,
Calmar ratios, Maximum Drawdown, and CVaR calculations.
"""

import logging
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RISK_FREE_RATE, ROLLING_SHARPE_WINDOW, ROLLING_MAX_DD_WINDOW

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Comprehensive risk metrics calculator for portfolio analysis.
    
    Provides calculation of standard risk-adjusted performance metrics
    used in institutional portfolio management.
    """
    
    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        """
        Initialize the risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"RiskMetrics initialized with risk_free_rate={risk_free_rate}")
    
    def calculate_all_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all risk metrics for a returns series.
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Override annual risk-free rate
            
        Returns:
            Dictionary containing all calculated metrics
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Convert to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        # Remove any NaN values
        returns = returns.dropna()
        
        if len(returns) < 2:
            logger.warning("Insufficient returns data for metrics calculation")
            return self._empty_metrics()
        
        # Calculate base statistics
        ann_return = self._annualized_return(returns)
        ann_vol = self._annualized_volatility(returns)
        
        # Calculate all metrics
        metrics = {
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': self._sharpe_ratio(ann_return, ann_vol, rf),
            'sortino_ratio': self._sortino_ratio(returns, rf),
            'max_drawdown': self._max_drawdown(returns),
            'calmar_ratio': self._calmar_ratio(returns, rf),
            'cvar_95': self._cvar(returns, alpha=0.05),
            'var_95': self._var(returns, alpha=0.05),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'positive_days_pct': float((returns > 0).mean()),
            'avg_win': float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0,
            'avg_loss': float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0,
        }
        
        # Win/loss ratio
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0.0
        
        logger.info(f"Calculated all metrics: Sharpe={metrics['sharpe_ratio']:.3f}, MaxDD={metrics['max_drawdown']:.2%}")
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return dictionary with NaN values for all metrics."""
        return {
            'annualized_return': np.nan,
            'annualized_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'sortino_ratio': np.nan,
            'max_drawdown': np.nan,
            'calmar_ratio': np.nan,
            'cvar_95': np.nan,
            'var_95': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'positive_days_pct': np.nan,
            'avg_win': np.nan,
            'avg_loss': np.nan,
            'win_loss_ratio': np.nan,
        }
    
    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return from daily returns."""
        return float(returns.mean() * 252)
    
    def _annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility from daily returns."""
        return float(returns.std() * np.sqrt(252))
    
    def _sharpe_ratio(
        self,
        ann_return: float,
        ann_volatility: float,
        risk_free_rate: float
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe = (Annualized Return - Risk Free Rate) / Annualized Volatility
        """
        if ann_volatility == 0 or np.isnan(ann_volatility):
            return 0.0
        return (ann_return - risk_free_rate) / ann_volatility
    
    def _sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """
        Calculate Sortino ratio using downside deviation.
        
        Sortino = (Annualized Return - Risk Free Rate) / Downside Deviation
        """
        ann_return = self._annualized_return(returns)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if ann_return > risk_free_rate else 0.0
        
        downside_std = float(downside_returns.std() * np.sqrt(252))
        
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        
        return (ann_return - risk_free_rate) / downside_std
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from peak.
        
        Max DD = (Peak - Trough) / Peak
        """
        # Compute cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Compute running maximum
        rolling_max = cum_returns.expanding().max()
        
        # Compute drawdown series
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        # Return maximum drawdown (most negative value)
        return float(drawdowns.min())
    
    def _calmar_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """
        Calculate Calmar ratio.
        
        Calmar = Annualized Return / |Max Drawdown|
        """
        ann_return = self._annualized_return(returns)
        max_dd = abs(self._max_drawdown(returns))
        
        if max_dd == 0 or np.isnan(max_dd):
            return 0.0
        
        return ann_return / max_dd
    
    def _var(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """
        Calculate Value at Risk at given alpha level.
        
        VaR = Returns at alpha percentile
        """
        return float(returns.quantile(alpha))
    
    def _cvar(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        CVaR = Mean of returns below VaR threshold
        """
        var_threshold = self._var(returns, alpha)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return float(tail_returns.mean())
    
    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = ROLLING_SHARPE_WINDOW,
        risk_free_rate: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Series of daily returns
            window: Rolling window in trading days
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Series of rolling Sharpe ratios
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rf_daily = rf / 252
        
        # Calculate rolling mean and std
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        rolling_std = returns.rolling(window=window, min_periods=window).std()
        
        # Annualize
        rolling_sharpe = ((rolling_mean - rf_daily) / rolling_std) * np.sqrt(252)
        
        return rolling_sharpe
    
    def calculate_rolling_max_drawdown(
        self,
        returns: pd.Series,
        window: int = ROLLING_MAX_DD_WINDOW
    ) -> pd.Series:
        """
        Calculate rolling maximum drawdown.
        
        Args:
            returns: Series of daily returns
            window: Rolling window in trading days
            
        Returns:
            Series of rolling maximum drawdowns
        """
        # Compute cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        def rolling_dd(x):
            """Calculate max drawdown for a window."""
            if len(x) < 2:
                return np.nan
            peak = x.expanding().max()
            dd = (x - peak) / peak
            return dd.min()
        
        rolling_max_dd = cum_returns.rolling(window=window, min_periods=window).apply(
            rolling_dd, raw=False
        )
        
        return rolling_max_dd
    
    def calculate_drawdown_series(self, returns: pd.Series) -> pd.DataFrame:
        """
        Calculate complete drawdown series with peak and recovery dates.
        
        Args:
            returns: Series of daily returns
            
        Returns:
            DataFrame with drawdown details
        """
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        df = pd.DataFrame({
            'cumulative_return': cum_returns,
            'peak': rolling_max,
            'drawdown': drawdowns
        })
        
        return df
