"""
Walk-Forward Backtesting Engine

This module implements institutional-grade walk-forward backtesting with
regime detection, CVaR optimization, and anti-leakage guarantees.
"""

import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    INITIAL_CAPITAL,
    PORTFOLIO_ASSETS,
    TRANSACTION_COST_RATE,
    SLIPPAGE_RATE,
    TEST_START,
    TEST_END
)
from src.portfolio.cvar_optimizer import CVaROptimizer
from src.models.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-Forward Backtesting Engine with strict anti-leakage.
    
    This engine performs true out-of-sample backtesting by:
    1. Only using features available at prediction time
    2. Detecting regime using HMM on historical data
    3. Predicting quintiles using regime-specific XGBoost models
    4. Optimizing portfolio using CVaR with regime constraints
    5. Tracking performance with realistic transaction costs
    
    Attributes:
        features_df: DataFrame of features (indexed by date)
        returns_df: DataFrame of asset returns (indexed by date)
        regime_detector: Trained HMM regime detector
        ensemble: Dictionary of regime-specific XGBoost models
        initial_capital: Starting portfolio value
    """
    
    HMM_FEATURES = [
        'log_returns', 'parkinson_vol', 'rolling_sharpe', 'rolling_max_dd',
        'vix_level', 'vix_percentile', 'rsi', 'momentum_20d'
    ]
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        regime_detector: RegimeDetector,
        ensemble: Dict,
        initial_capital: float = INITIAL_CAPITAL
    ):
        """
        Initialize the walk-forward backtest.
        
        Args:
            features_df: Feature data with DatetimeIndex
            returns_df: Asset returns with DatetimeIndex
            regime_detector: Trained HMM model
            ensemble: Dictionary with 'models' and 'feature_names' keys
            initial_capital: Starting portfolio value
        """
        self.features_df = features_df
        self.returns_df = returns_df
        self.regime_detector = regime_detector
        self.ensemble = ensemble
        self.initial_capital = initial_capital
        
        # Results storage
        self.results = []
        self.trade_log = []
        self.regime_transitions = []
        
        # Current state
        self.current_weights = {asset: 1.0/len(PORTFOLIO_ASSETS) for asset in PORTFOLIO_ASSETS}
        self.portfolio_value = initial_capital
        self.current_regime = None
        
        logger.info(f"WalkForwardBacktest initialized with {initial_capital:,.2f} capital")
    
    def run(
        self,
        start_date: str = TEST_START,
        end_date: str = TEST_END,
        lookback_days: int = 60
    ) -> pd.DataFrame:
        """
        Run the walk-forward backtest.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            lookback_days: Number of days for CVaR estimation
            
        Returns:
            DataFrame with backtest results
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Get weekly rebalancing dates
        test_features = self.features_df.loc[start:end]
        test_returns = self.returns_df.loc[start:end]
        
        if len(test_features) == 0:
            raise ValueError(f"No data found for period {start_date} to {end_date}")
        
        logger.info(f"Starting backtest from {start_date} to {end_date} ({len(test_features)} periods)")
        
        # Track benchmark (SPY buy-and-hold)
        benchmark_value = self.initial_capital
        cumulative_max = self.initial_capital
        
        # Iterate through each week
        for i, (date, features_row) in enumerate(test_features.iterrows()):
            # 1. Detect regime (using only past data)
            regime = self._detect_regime(features_row)
            
            # Check for regime transition
            if self.current_regime is not None and regime != self.current_regime:
                self.regime_transitions.append({
                    'date': date,
                    'from_regime': self.current_regime,
                    'to_regime': regime
                })
                logger.info(f"Regime transition: {self.current_regime} -> {regime} on {date}")
            
            self.current_regime = regime
            
            # 2. Predict quintile using regime-specific model
            quintile, confidence = self._predict_quintile(features_row, regime)
            
            # 3. Get optimal weights from CVaR optimizer
            historical_returns = self.returns_df.loc[:date].tail(lookback_days)
            if len(historical_returns) < 20:
                new_weights = self.current_weights
            else:
                new_weights = self._optimize_portfolio(historical_returns, regime, quintile, confidence)
            
            # 4. Calculate transaction costs if rebalancing
            transaction_cost = self._calculate_transaction_cost(self.current_weights, new_weights)
            
            # 5. Apply portfolio return for this period
            if date in test_returns.index:
                returns_row = test_returns.loc[date]
                portfolio_return = sum(self.current_weights[asset] * returns_row[asset] 
                                       for asset in PORTFOLIO_ASSETS if asset in returns_row.index)
                benchmark_return = returns_row.get('SPY', 0)
            else:
                portfolio_return = 0
                benchmark_return = 0
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= transaction_cost
            benchmark_value *= (1 + benchmark_return)
            
            # Update cumulative max for drawdown
            cumulative_max = max(cumulative_max, self.portfolio_value)
            drawdown = (self.portfolio_value - cumulative_max) / cumulative_max
            
            # Log trade if weights changed
            if transaction_cost > 0:
                self.trade_log.append({
                    'date': date,
                    'old_weights': self.current_weights.copy(),
                    'new_weights': new_weights.copy(),
                    'transaction_cost': transaction_cost,
                    'regime': regime
                })
            
            # Store results
            self.results.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'benchmark_value': benchmark_value,
                'weekly_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'regime': regime,
                'regime_conf': confidence,
                'quintile_pred': quintile,
                'spy_weight': new_weights.get('SPY', 0),
                'tlt_weight': new_weights.get('TLT', 0),
                'gld_weight': new_weights.get('GLD', 0),
                'drawdown': drawdown,
                'transaction_cost': transaction_cost
            })
            
            # Update current weights
            self.current_weights = new_weights.copy()
            
            # Progress logging
            if (i + 1) % 13 == 0:
                logger.info(f"Progress: {i+1}/{len(test_features)} weeks | "
                           f"Portfolio: ${self.portfolio_value:,.0f} | "
                           f"Benchmark: ${benchmark_value:,.0f} | Regime: {regime}")
        
        logger.info(f"Backtest complete. Final portfolio: ${self.portfolio_value:,.2f}")
        return pd.DataFrame(self.results)
    
    def _detect_regime(self, features_row: pd.Series) -> int:
        """Detect market regime using HMM."""
        try:
            # Use model's expected features if available
            if hasattr(self.regime_detector, 'feature_names') and self.regime_detector.feature_names:
                required_features = self.regime_detector.feature_names
            else:
                required_features = self.HMM_FEATURES
            
            hmm_features = [f for f in required_features if f in features_row.index]
            
            # Check for feature mismatch
            if len(hmm_features) != len(required_features):
                missing = set(required_features) - set(features_row.index)
                if missing:
                    logger.warning(f"Missing features for regime detection: {missing}. Defaulting to 1")
                    return 1
            
            # Extract values in correct order
            feature_values = features_row[required_features].values.reshape(1, -1)
            regime = self.regime_detector.predict(feature_values)[0]
            return int(regime)
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Defaulting to 1")
            return 1
    
    def _predict_quintile(self, features_row: pd.Series, regime: int) -> Tuple[int, float]:
        """Predict return quintile using regime-specific XGBoost model."""
        try:
            models = self.ensemble.get('models', {})
            feature_names = self.ensemble.get('feature_names', [])
            
            model = models.get(regime)
            if model is None:
                return 3, 0.5  # Default neutral
            
            # Get feature vector
            available_features = [f for f in feature_names if f in features_row.index]
            if len(available_features) < len(feature_names) * 0.5:
                return 3, 0.5
            
            feature_values = features_row[available_features].values.reshape(1, -1)
            
            # Predict quintile
            quintile = int(model.predict(feature_values)[0])
            quintile = max(1, min(5, quintile))  # Clamp to 1-5
            
            # Get prediction probabilities if available
            try:
                proba = model.predict_proba(feature_values)
                confidence = float(proba.max())
            except:
                confidence = 0.5
            
            return quintile, confidence
        except Exception as e:
            logger.warning(f"Quintile prediction failed: {e}. Defaulting to neutral")
            return 3, 0.5
    
    def _optimize_portfolio(
        self,
        historical_returns: pd.DataFrame,
        regime: int,
        quintile: int,
        confidence: float
    ) -> Dict[str, float]:
        """Optimize portfolio weights using CVaR."""
        try:
            optimizer = CVaROptimizer(historical_returns, regime)
            weights = optimizer.optimize()
            weights = optimizer.adjust_for_confidence(weights, quintile, confidence, regime)
            return weights
        except Exception as e:
            logger.warning(f"CVaR optimization failed: {e}. Using heuristic weights")
            return CVaROptimizer.HEURISTIC_WEIGHTS.get(regime, {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2})
    
    def _calculate_transaction_cost(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """Calculate transaction costs for rebalancing."""
        turnover = sum(abs(new_weights.get(a, 0) - old_weights.get(a, 0)) for a in PORTFOLIO_ASSETS)
        cost = turnover * self.portfolio_value * (TRANSACTION_COST_RATE + SLIPPAGE_RATE)
        return cost
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get the trade log as a DataFrame."""
        return pd.DataFrame(self.trade_log)
    
    def get_regime_transitions(self) -> pd.DataFrame:
        """Get regime transitions as a DataFrame."""
        return pd.DataFrame(self.regime_transitions)
