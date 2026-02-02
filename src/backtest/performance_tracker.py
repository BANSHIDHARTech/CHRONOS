"""
Backtest Performance Tracker

This module tracks portfolio performance metrics during backtesting,
including returns, drawdowns, Sharpe ratio, and regime-specific analysis.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestPerformanceTracker:
    
    
    def __init__(self, results_df: pd.DataFrame, risk_free_rate: float = 0.02):
        
        self.results_df = results_df.copy()
        self.risk_free_rate = risk_free_rate
        
        
        if 'date' in self.results_df.columns:
            self.results_df.set_index('date', inplace=True)
        
        logger.info(f"PerformanceTracker initialized with {len(results_df)} periods")
    
    def calculate_summary_statistics(self) -> Dict:
        
        df = self.results_df
        
        
        initial_value = df['portfolio_value'].iloc[0]
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        initial_benchmark = df['benchmark_value'].iloc[0]
        final_benchmark = df['benchmark_value'].iloc[-1]
        benchmark_return = (final_benchmark / initial_benchmark) - 1
        
        
        weekly_returns = df['weekly_return']
        benchmark_weekly = df['benchmark_return'] if 'benchmark_return' in df.columns else pd.Series([0])
        
        
        annualization_factor = 52
        
        annualized_return = (1 + total_return) ** (annualization_factor / len(df)) - 1
        annualized_volatility = weekly_returns.std() * np.sqrt(annualization_factor)
        
        annualized_benchmark_return = (1 + benchmark_return) ** (annualization_factor / len(df)) - 1
        annualized_benchmark_vol = benchmark_weekly.std() * np.sqrt(annualization_factor)
        
       
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        benchmark_sharpe = (annualized_benchmark_return - self.risk_free_rate) / annualized_benchmark_vol if annualized_benchmark_vol > 0 else 0
        
        
        max_dd_chronos = df['drawdown'].min()
        
        
        cummax_benchmark = df['benchmark_value'].cummax()
        benchmark_dd = (df['benchmark_value'] - cummax_benchmark) / cummax_benchmark
        max_dd_benchmark = benchmark_dd.min()
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_dd_chronos) if max_dd_chronos != 0 else 0
        
        # Win Rate
        win_rate = (weekly_returns > 0).sum() / len(weekly_returns) if len(weekly_returns) > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = weekly_returns[weekly_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Information Ratio
        excess_returns = weekly_returns - benchmark_weekly
        tracking_error = excess_returns.std() * np.sqrt(annualization_factor)
        information_ratio = (annualized_return - annualized_benchmark_return) / tracking_error if tracking_error > 0 else 0
        
        return {
            
            'initial_value': initial_value,
            'final_value_chronos': final_value,
            'final_value_benchmark': final_benchmark,
            'total_return_chronos': total_return,
            'total_return_benchmark': benchmark_return,
            'outperformance': total_return - benchmark_return,
            
            
            'annualized_return_chronos': annualized_return,
            'annualized_return_benchmark': annualized_benchmark_return,
            'annualized_volatility_chronos': annualized_volatility,
            'annualized_volatility_benchmark': annualized_benchmark_vol,
            
            
            'sharpe_ratio_chronos': sharpe_ratio,
            'sharpe_ratio_benchmark': benchmark_sharpe,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            
            
            'max_dd_chronos': max_dd_chronos,
            'max_dd_benchmark': max_dd_benchmark,
            'dd_protection': max_dd_benchmark - max_dd_chronos,
            
            
            'win_rate': win_rate,
            'num_weeks': len(df),
            
            
            'risk_free_rate': self.risk_free_rate
        }
    
    def calculate_regime_performance(self) -> pd.DataFrame:
        
        df = self.results_df
        
        regime_stats = []
        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime]
            
            if len(regime_df) > 0:
                regime_stats.append({
                    'regime': regime,
                    'num_weeks': len(regime_df),
                    'pct_time': len(regime_df) / len(df),
                    'avg_return': regime_df['weekly_return'].mean(),
                    'total_return': (1 + regime_df['weekly_return']).prod() - 1,
                    'volatility': regime_df['weekly_return'].std(),
                    'sharpe': regime_df['weekly_return'].mean() / regime_df['weekly_return'].std() if regime_df['weekly_return'].std() > 0 else 0,
                    'max_drawdown': regime_df['drawdown'].min(),
                    'win_rate': (regime_df['weekly_return'] > 0).sum() / len(regime_df)
                })
        
        return pd.DataFrame(regime_stats)
    
    def calculate_rolling_metrics(self, window: int = 13) -> pd.DataFrame:
        
        df = self.results_df.copy()
        
        df['rolling_return'] = df['weekly_return'].rolling(window).sum()
        df['rolling_volatility'] = df['weekly_return'].rolling(window).std() * np.sqrt(52)
        df['rolling_sharpe'] = df['rolling_return'] * (52/window) / df['rolling_volatility']
        
        
        rolling_max = df['portfolio_value'].rolling(window).max()
        df['rolling_drawdown'] = (df['portfolio_value'] - rolling_max) / rolling_max
        
        return df[['rolling_return', 'rolling_volatility', 'rolling_sharpe', 'rolling_drawdown']]
    
    def calculate_attribution(self) -> pd.DataFrame:
        
        df = self.results_df
        
        
        attributions = []
        for asset in ['SPY', 'TLT', 'GLD']:
            weight_col = f'{asset.lower()}_weight'
            if weight_col in df.columns:
                avg_weight = df[weight_col].mean()
                
                attributions.append({
                    'asset': asset,
                    'avg_weight': avg_weight,
                    'estimated_contribution': avg_weight  
                })
        
        return pd.DataFrame(attributions)
    
    def get_drawdown_periods(self) -> pd.DataFrame:
        
        df = self.results_df.copy()
        
        
        in_drawdown = False
        dd_start = None
        drawdown_periods = []
        
        for date, row in df.iterrows():
            if row['drawdown'] < -0.02 and not in_drawdown:
                in_drawdown = True
                dd_start = date
            elif row['drawdown'] >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    'start': dd_start,
                    'end': date,
                    'max_drawdown': df.loc[dd_start:date, 'drawdown'].min()
                })
        
        return pd.DataFrame(drawdown_periods)
