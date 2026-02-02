"""
CVaR (Conditional Value at Risk) Portfolio Optimizer

This module implements institutional-grade CVaR optimization with regime-specific
constraints. CVaR minimizes expected loss in the worst 5% of scenarios, making it
superior to traditional mean-variance optimization for tail risk management.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pypfopt import EfficientCVaR
from pypfopt import expected_returns

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    PORTFOLIO_ASSETS,
    REGIME_CONSTRAINTS,
    HIGH_CONFIDENCE_THRESHOLD,
    BULLISH_QUINTILE,
    BEARISH_QUINTILE,
    EQUITY_BOOST_FACTOR,
    EQUITY_CUT_FACTOR
)

logger = logging.getLogger(__name__)


class CVaROptimizer:
    """
    CVaR Portfolio Optimizer with regime-specific constraints.
    
    This optimizer uses Conditional Value at Risk (Expected Shortfall) to
    minimize tail risk while respecting regime-based allocation bounds.
    
    Attributes:
        returns: DataFrame of historical returns for portfolio assets
        regime: Current market regime (0=Euphoria, 1=Complacency, 2=Capitulation)
        assets: List of asset tickers in portfolio
    """
    
    # Heuristic fallback weights when optimization fails
    HEURISTIC_WEIGHTS = {
        0: {'SPY': 0.70, 'TLT': 0.20, 'GLD': 0.10},  # Euphoria
        1: {'SPY': 0.50, 'TLT': 0.30, 'GLD': 0.20},  # Complacency
        2: {'SPY': 0.10, 'TLT': 0.60, 'GLD': 0.30},  # Capitulation
    }
    
    def __init__(self, returns: pd.DataFrame, regime: int):
        """
        Initialize the CVaR optimizer.
        
        Args:
            returns: DataFrame with columns for each asset, containing historical returns
            regime: Current market regime (0, 1, or 2)
        """
        self.returns = returns
        self.regime = regime
        self.assets = PORTFOLIO_ASSETS
        
        # Validate inputs
        if regime not in [0, 1, 2]:
            raise ValueError(f"Invalid regime: {regime}. Must be 0, 1, or 2.")
        
        missing_assets = [a for a in self.assets if a not in returns.columns]
        if missing_assets:
            raise ValueError(f"Missing assets in returns data: {missing_assets}")
        
        # Subset returns to portfolio assets only
        self.returns = self.returns[self.assets]
        
        logger.info(f"CVaROptimizer initialized for regime {regime} with {len(self.returns)} observations")
    
    def optimize(self) -> Dict[str, float]:
        """
        Perform CVaR optimization with regime-specific constraints.
        
        Returns:
            Dictionary mapping asset names to optimal weights
        """
        try:
            # Calculate annualized expected returns (mean × 252)
            mu = expected_returns.mean_historical_return(
                self.returns,
                returns_data=True,
                frequency=252
            )
            
            # Initialize EfficientCVaR optimizer
            ef_cvar = EfficientCVaR(
                mu,
                self.returns,
                beta=0.95  # 95% CVaR (focus on worst 5% of returns)
            )
            
            # Apply regime-specific constraints
            self._apply_regime_constraints(ef_cvar)
            
            # Minimize CVaR (Conditional Value at Risk)
            ef_cvar.min_cvar()
            
            # Get cleaned weights (removes tiny allocations < 1e-4)
            weights = ef_cvar.clean_weights()
            
            logger.info(f"CVaR optimization successful. Weights: {weights}")
            return dict(weights)
            
        except Exception as e:
            logger.warning(f"CVaR optimization failed: {e}. Falling back to heuristic weights.")
            return self.get_heuristic_weights(self.regime)
    
    def _apply_regime_constraints(self, ef_cvar: EfficientCVaR) -> None:
        """
        Apply regime-specific min/max constraints to each asset.
        
        Args:
            ef_cvar: EfficientCVaR optimizer instance
        """
        constraints = REGIME_CONSTRAINTS.get(self.regime, {})
        
        # Closure factory to properly capture loop variables
        def make_min_constraint(idx, min_w):
            return lambda w: w[idx] >= min_w
        
        def make_max_constraint(idx, max_w):
            return lambda w: w[idx] <= max_w
        
        for i, asset in enumerate(self.assets):
            if asset in constraints:
                min_weight, max_weight = constraints[asset]
                
                # Add minimum weight constraint (fixed closure bug)
                ef_cvar.add_constraint(make_min_constraint(i, min_weight))
                
                # Add maximum weight constraint (fixed closure bug)
                ef_cvar.add_constraint(make_max_constraint(i, max_weight))
                
                logger.debug(f"Applied constraints for {asset}: [{min_weight}, {max_weight}]")
    
    def get_heuristic_weights(self, regime: int) -> Dict[str, float]:
        """
        Get fallback heuristic weights for a given regime.
        
        Args:
            regime: Market regime (0, 1, or 2)
            
        Returns:
            Dictionary of heuristic weights
        """
        return self.HEURISTIC_WEIGHTS.get(regime, self.HEURISTIC_WEIGHTS[1])
    
    def adjust_for_confidence(
        self,
        base_weights: Dict[str, float],
        predicted_quintile: int,
        prediction_confidence: float,
        current_regime: int
    ) -> Dict[str, float]:
        """
        Adjust portfolio weights based on prediction confidence.
        
        High confidence bullish predictions boost equity allocation, while
        high confidence bearish predictions reduce it.
        
        Args:
            base_weights: Base weights from CVaR optimization
            predicted_quintile: Predicted return quintile (1-5)
            prediction_confidence: Confidence score (0-1)
            current_regime: Current market regime
            
        Returns:
            Adjusted and renormalized weights
        """
        weights = base_weights.copy()
        constraints = REGIME_CONSTRAINTS.get(current_regime, {})
        
        spy_min, spy_max = constraints.get('SPY', (0.0, 1.0))
        
        # High confidence bullish: increase equity
        if predicted_quintile == BULLISH_QUINTILE and prediction_confidence > HIGH_CONFIDENCE_THRESHOLD:
            weights['SPY'] = min(weights['SPY'] * EQUITY_BOOST_FACTOR, spy_max)
            logger.info(f"Bullish confidence adjustment: SPY weight boosted to {weights['SPY']:.2%}")
        
        # High confidence bearish: decrease equity
        elif predicted_quintile == BEARISH_QUINTILE and prediction_confidence > HIGH_CONFIDENCE_THRESHOLD:
            weights['SPY'] = max(weights['SPY'] * EQUITY_CUT_FACTOR, spy_min)
            logger.info(f"Bearish confidence adjustment: SPY weight reduced to {weights['SPY']:.2%}")
        
        # Renormalize weights to sum to 1.0
        weights = self._renormalize_weights(weights)
        
        return weights
    
    def _renormalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Renormalize weights to ensure they sum to 1.0.
        
        Args:
            weights: Dictionary of weights that may not sum to 1
            
        Returns:
            Renormalized weights summing to 1.0
        """
        total = sum(weights.values())
        if total == 0:
            # Equal weight fallback
            return {k: 1.0 / len(weights) for k in weights}
        
        return {k: v / total for k, v in weights.items()}
