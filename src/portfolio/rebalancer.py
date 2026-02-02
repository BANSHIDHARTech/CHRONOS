"""
Portfolio Rebalancing Engine

This module handles rebalancing logic with realistic transaction costs and
slippage modeling. Designed for institutional-grade backtesting accuracy.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    TRANSACTION_COST_RATE,
    SLIPPAGE_RATE,
    REBALANCE_FREQUENCY
)

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine with transaction cost modeling.
    
    Handles the mechanics of portfolio rebalancing including:
    - Rebalancing frequency enforcement
    - Turnover calculation
    - Transaction cost and slippage application
    
    Attributes:
        transaction_cost_rate: Cost per trade as fraction (default 10bps)
        slippage_rate: Market impact cost as fraction (default 5bps)
    """
    
    # Frequency thresholds in days
    FREQUENCY_DAYS = {
        'weekly': 7,
        'bi-weekly': 14,
        'monthly': 30,
        'daily': 1,
    }
    
    def __init__(
        self,
        transaction_cost_rate: float = TRANSACTION_COST_RATE,
        slippage_rate: float = SLIPPAGE_RATE
    ):
        """
        Initialize the rebalancer.
        
        Args:
            transaction_cost_rate: Transaction cost as fraction (default 0.001 = 10bps)
            slippage_rate: Slippage cost as fraction (default 0.0005 = 5bps)
        """
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_rate = slippage_rate
        
        logger.info(
            f"PortfolioRebalancer initialized with "
            f"transaction_cost={transaction_cost_rate:.4f}, "
            f"slippage={slippage_rate:.4f}"
        )
    
    def should_rebalance(
        self,
        current_date: datetime,
        last_rebalance_date: Optional[datetime],
        frequency: str = REBALANCE_FREQUENCY
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_date: Current trading date
            last_rebalance_date: Date of last rebalance (None if never rebalanced)
            frequency: Rebalancing frequency ('daily', 'weekly', 'bi-weekly', 'monthly')
            
        Returns:
            True if rebalancing should occur
        """
        # Always rebalance on first call
        if last_rebalance_date is None:
            return True
        
        # Get frequency threshold
        frequency_days = self.FREQUENCY_DAYS.get(frequency.lower(), 7)
        
        # Check if enough time has elapsed
        days_elapsed = (current_date - last_rebalance_date).days
        
        should_rebal = days_elapsed >= frequency_days
        
        if should_rebal:
            logger.debug(
                f"Rebalancing triggered: {days_elapsed} days since last rebalance "
                f"(threshold: {frequency_days})"
            )
        
        return should_rebal
    
    def calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio turnover between two weight allocations.
        
        Turnover is defined as the sum of absolute weight changes across all assets.
        A 100% turnover means completely new portfolio; 0% means no changes.
        
        Args:
            old_weights: Current portfolio weights
            new_weights: Target portfolio weights
            
        Returns:
            Turnover as a fraction (0.0 to 2.0)
        """
        all_assets = set(old_weights.keys()) | set(new_weights.keys())
        
        turnover = sum(
            abs(new_weights.get(asset, 0.0) - old_weights.get(asset, 0.0))
            for asset in all_assets
        )
        
        logger.debug(f"Portfolio turnover: {turnover:.4f}")
        return turnover
    
    def apply_transaction_costs(
        self,
        portfolio_value: float,
        turnover: float,
        transaction_cost_rate: Optional[float] = None,
        slippage_rate: Optional[float] = None
    ) -> float:
        """
        Apply transaction costs and slippage to portfolio value.
        
        Args:
            portfolio_value: Current portfolio value
            turnover: Portfolio turnover as fraction
            transaction_cost_rate: Override transaction cost rate
            slippage_rate: Override slippage rate
            
        Returns:
            Adjusted portfolio value after costs
        """
        tc_rate = transaction_cost_rate or self.transaction_cost_rate
        slip_rate = slippage_rate or self.slippage_rate
        
        total_cost_rate = tc_rate + slip_rate
        total_cost = portfolio_value * turnover * total_cost_rate
        
        adjusted_value = portfolio_value - total_cost
        
        logger.debug(
            f"Transaction costs: ${total_cost:.2f} "
            f"({turnover:.2%} turnover × {total_cost_rate:.4%} rate)"
        )
        
        return adjusted_value
    
    def rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        returns_today: Dict[str, float]
    ) -> Tuple[float, Dict[str, float], float]:
        """
        Execute portfolio rebalance with transaction costs.
        
        This method:
        1. Calculates turnover from current to target weights
        2. Applies transaction costs and slippage
        3. Updates portfolio value based on today's returns
        4. Returns new portfolio state
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights after rebalance
            portfolio_value: Current portfolio value before rebalance
            returns_today: Today's returns for each asset
            
        Returns:
            Tuple of (new_portfolio_value, new_weights, transaction_cost)
        """
        # Calculate turnover
        turnover = self.calculate_turnover(current_weights, target_weights)
        
        # Apply transaction costs to get value after rebalancing costs
        total_cost_rate = self.transaction_cost_rate + self.slippage_rate
        transaction_cost = portfolio_value * turnover * total_cost_rate
        value_after_costs = portfolio_value - transaction_cost
        
        # Calculate portfolio return for today using target weights
        portfolio_return = sum(
            target_weights.get(asset, 0.0) * returns_today.get(asset, 0.0)
            for asset in target_weights
        )
        
        # Update portfolio value based on today's returns
        new_portfolio_value = value_after_costs * (1 + portfolio_return)
        
        logger.info(
            f"Rebalance executed: turnover={turnover:.2%}, "
            f"cost=${transaction_cost:.2f}, "
            f"return={portfolio_return:.4%}, "
            f"new_value=${new_portfolio_value:.2f}"
        )
        
        return new_portfolio_value, target_weights.copy(), transaction_cost
    
    def calculate_drift(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        drift_threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Check if portfolio has drifted beyond threshold.
        
        Useful for drift-based rebalancing strategies.
        
        Args:
            target_weights: Target allocation weights
            current_weights: Current portfolio weights (after price movements)
            drift_threshold: Maximum allowed drift before rebalancing
            
        Returns:
            Tuple of (rebalance_needed, max_drift)
        """
        max_drift = max(
            abs(current_weights.get(asset, 0.0) - target_weights.get(asset, 0.0))
            for asset in target_weights
        )
        
        return max_drift > drift_threshold, max_drift
