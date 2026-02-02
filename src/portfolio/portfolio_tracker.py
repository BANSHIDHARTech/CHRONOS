"""
Portfolio Value and Attribution Tracker

This module tracks portfolio performance over time, including value history,
weight allocations, transaction costs, and performance attribution.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import INITIAL_CAPITAL, PORTFOLIO_ASSETS

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Portfolio value and attribution tracker.
    
    Tracks portfolio performance over time with detailed history of:
    - Portfolio values
    - Weight allocations
    - Daily returns
    - Transaction costs
    - Regime changes
    - Performance attribution by asset
    
    Attributes:
        initial_capital: Starting portfolio value
        current_value: Current portfolio value
        assets: List of tracked assets
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        assets: Optional[List[str]] = None
    ):
        """
        Initialize the portfolio tracker.
        
        Args:
            initial_capital: Starting portfolio value (default $100,000)
            assets: List of asset tickers to track
        """
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.assets = assets or PORTFOLIO_ASSETS
        
        # History tracking
        self.dates: List[datetime] = []
        self.portfolio_values: List[float] = []
        self.weights_history: List[Dict[str, float]] = []
        self.returns_history: List[float] = []
        self.transaction_costs_history: List[float] = []
        self.regime_history: List[int] = []
        self.asset_returns_history: List[Dict[str, float]] = []
        
        logger.info(f"PortfolioTracker initialized with ${initial_capital:,.2f} capital")
    
    def update(
        self,
        date: datetime,
        weights: Dict[str, float],
        returns_today: Dict[str, float],
        regime: int,
        transaction_cost: float = 0.0
    ) -> float:
        """
        Update portfolio with new day's data.
        
        Args:
            date: Current date
            weights: Current portfolio weights
            returns_today: Today's returns for each asset
            regime: Current market regime
            transaction_cost: Transaction costs incurred today
            
        Returns:
            New portfolio value
        """
        # Calculate portfolio return
        portfolio_return = sum(
            weights.get(asset, 0.0) * returns_today.get(asset, 0.0)
            for asset in self.assets
        )
        
        # Update portfolio value
        self.current_value = self.current_value * (1 + portfolio_return) - transaction_cost
        
        # Record history
        self.dates.append(date)
        self.portfolio_values.append(self.current_value)
        self.weights_history.append(weights.copy())
        self.returns_history.append(portfolio_return)
        self.transaction_costs_history.append(transaction_cost)
        self.regime_history.append(regime)
        self.asset_returns_history.append(returns_today.copy())
        
        logger.debug(
            f"Portfolio updated: date={date.strftime('%Y-%m-%d')}, "
            f"return={portfolio_return:.4%}, value=${self.current_value:,.2f}"
        )
        
        return self.current_value
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get portfolio performance summary as DataFrame.
        
        Returns:
            DataFrame with columns: date, portfolio_value, regime, 
            and weight columns for each asset
        """
        if not self.dates:
            return pd.DataFrame()
        
        # Build base DataFrame
        df = pd.DataFrame({
            'date': self.dates,
            'portfolio_value': self.portfolio_values,
            'regime': self.regime_history,
            'daily_return': self.returns_history,
            'transaction_cost': self.transaction_costs_history,
        })
        
        # Add weight columns for each asset
        for asset in self.assets:
            df[f'{asset}_weight'] = [
                w.get(asset, 0.0) for w in self.weights_history
            ]
        
        df.set_index('date', inplace=True)
        
        return df
    
    def calculate_attribution(self) -> pd.DataFrame:
        """
        Calculate performance attribution by asset.
        
        Decomposes portfolio return into contributions from each asset.
        
        Returns:
            DataFrame with columns: date and contribution columns for each asset
        """
        if not self.dates:
            return pd.DataFrame()
        
        attribution_data = []
        
        for i, date in enumerate(self.dates):
            weights = self.weights_history[i]
            asset_returns = self.asset_returns_history[i]
            
            row = {'date': date}
            
            for asset in self.assets:
                weight = weights.get(asset, 0.0)
                ret = asset_returns.get(asset, 0.0)
                row[f'{asset}_contribution'] = weight * ret
            
            row['total_return'] = self.returns_history[i]
            attribution_data.append(row)
        
        df = pd.DataFrame(attribution_data)
        df.set_index('date', inplace=True)
        
        return df
    
    def get_cumulative_attribution(self) -> pd.DataFrame:
        """
        Calculate cumulative performance attribution.
        
        Returns:
            DataFrame with cumulative contributions for each asset
        """
        daily_attr = self.calculate_attribution()
        
        if daily_attr.empty:
            return pd.DataFrame()
        
        # Calculate cumulative contributions
        cumulative = daily_attr.cumsum()
        
        return cumulative
    
    def get_regime_performance(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate performance metrics by regime.
        
        Returns:
            Dictionary mapping regime to performance metrics
        """
        if not self.dates:
            return {}
        
        df = self.get_performance_summary()
        
        regime_perf = {}
        
        for regime in sorted(df['regime'].unique()):
            regime_df = df[df['regime'] == regime]
            returns = regime_df['daily_return']
            
            regime_perf[regime] = {
                'count': len(regime_df),
                'mean_return': float(returns.mean()),
                'total_return': float((1 + returns).prod() - 1),
                'volatility': float(returns.std()),
                'sharpe': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0,
            }
        
        return regime_perf
    
    def get_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns for reporting.
        
        Returns:
            Series of monthly returns
        """
        df = self.get_performance_summary()
        
        if df.empty:
            return pd.Series()
        
        # Convert to monthly
        df.index = pd.to_datetime(df.index)
        monthly = df['daily_return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return monthly
    
    def get_total_return(self) -> float:
        """Calculate total portfolio return since inception."""
        return (self.current_value - self.initial_capital) / self.initial_capital
    
    def get_total_transaction_costs(self) -> float:
        """Get total transaction costs paid."""
        return sum(self.transaction_costs_history)
    
    def reset(self):
        """Reset tracker to initial state."""
        self.current_value = self.initial_capital
        self.dates = []
        self.portfolio_values = []
        self.weights_history = []
        self.returns_history = []
        self.transaction_costs_history = []
        self.regime_history = []
        self.asset_returns_history = []
        
        logger.info("PortfolioTracker reset to initial state")
    
    def to_dict(self) -> Dict:
        """
        Export tracker state to dictionary for serialization.
        
        Returns:
            Dictionary with all tracker state
        """
        return {
            'initial_capital': self.initial_capital,
            'current_value': self.current_value,
            'assets': self.assets,
            'dates': [d.isoformat() if isinstance(d, datetime) else str(d) for d in self.dates],
            'portfolio_values': self.portfolio_values,
            'weights_history': self.weights_history,
            'returns_history': self.returns_history,
            'transaction_costs_history': self.transaction_costs_history,
            'regime_history': self.regime_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioTracker':
        """
        Create tracker from dictionary.
        
        Args:
            data: Dictionary with tracker state
            
        Returns:
            New PortfolioTracker instance
        """
        tracker = cls(
            initial_capital=data['initial_capital'],
            assets=data.get('assets')
        )
        
        tracker.current_value = data['current_value']
        tracker.dates = [datetime.fromisoformat(d) for d in data['dates']]
        tracker.portfolio_values = data['portfolio_values']
        tracker.weights_history = data['weights_history']
        tracker.returns_history = data['returns_history']
        tracker.transaction_costs_history = data['transaction_costs_history']
        tracker.regime_history = data['regime_history']
        
        return tracker
