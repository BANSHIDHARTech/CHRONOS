"""
Crisis Period Analyzer

This module analyzes portfolio performance during defined crisis periods,
comparing CHRONOS strategy performance against benchmarks.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import TEST_START, TEST_END

logger = logging.getLogger(__name__)



CRISIS_PERIODS = {
    'covid_crash_2020': {
        'name': 'COVID-19 Market Crash',
        'start': '2020-02-20',
        'end': '2020-03-23',
        'description': 'Fastest 30% decline in history due to pandemic fears'
    },
    'covid_recovery_volatility': {
        'name': 'COVID Recovery Volatility',
        'start': '2020-03-24',
        'end': '2020-06-08',
        'description': 'V-shaped recovery with extreme volatility'
    },
    'inflation_shock_2022': {
        'name': 'Inflation Shock 2022',
        'start': '2022-01-01',
        'end': '2022-06-17',
        'description': 'Bear market driven by inflation and Fed rate hikes'
    },
    'banking_crisis_2023': {
        'name': 'Regional Banking Crisis',
        'start': '2023-03-08',
        'end': '2023-03-20',
        'description': 'SVB collapse and regional bank contagion fears'
    },
    'rate_hike_selloff_2024': {
        'name': 'Rate Hike Selloff',
        'start': '2024-04-01',
        'end': '2024-04-19',
        'description': 'Market pullback on persistent inflation concerns'
    },
    'august_volatility_2024': {
        'name': 'August 2024 Volatility',
        'start': '2024-08-01',
        'end': '2024-08-12',
        'description': 'VIX spike and carry trade unwind'
    }
}


class CrisisAnalyzer:
    
    
    def __init__(self, results_df: pd.DataFrame, test_start: str = TEST_START, test_end: str = TEST_END):
        
        self.results_df = results_df.copy()
        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)
        
        # Ensure datetime index
        if 'date' in self.results_df.columns:
            self.results_df.set_index('date', inplace=True)
        
        if not isinstance(self.results_df.index, pd.DatetimeIndex):
            self.results_df.index = pd.to_datetime(self.results_df.index)
        
        logger.info(f"CrisisAnalyzer initialized for period {test_start} to {test_end}")
    
    def analyze_all_crises(self) -> pd.DataFrame:
        
        results = []
        
        for crisis_id, crisis_info in CRISIS_PERIODS.items():
            analysis = self.analyze_crisis_period(crisis_id, crisis_info)
            results.append(analysis)
        
        df = pd.DataFrame(results)
        logger.info(f"Analyzed {len(results)} crisis periods")
        return df
    
    def analyze_crisis_period(self, crisis_id: str, crisis_info: Dict) -> Dict:
        
        start = pd.to_datetime(crisis_info['start'])
        end = pd.to_datetime(crisis_info['end'])
        
        result = {
            'crisis_id': crisis_id,
            'name': crisis_info['name'],
            'start': crisis_info['start'],
            'end': crisis_info['end'],
            'description': crisis_info['description'],
            'in_test_period': False
        }
        
        
        if end < self.test_start or start > self.test_end:
            result['in_test_period'] = False
            result['status'] = 'not_in_backtest_period'
            return result
        
        
        analysis_start = max(start, self.test_start)
        analysis_end = min(end, self.test_end)
        
        
        try:
            crisis_data = self.results_df.loc[analysis_start:analysis_end]
        except KeyError:
            result['status'] = 'no_data_available'
            return result
        
        if len(crisis_data) == 0:
            result['status'] = 'no_data_in_period'
            return result
        
        result['in_test_period'] = True
        result['num_periods'] = len(crisis_data)
        
        
        if 'portfolio_value' in crisis_data.columns:
            start_value = crisis_data['portfolio_value'].iloc[0]
            end_value = crisis_data['portfolio_value'].iloc[-1]
            result['chronos_return'] = (end_value / start_value) - 1
        
        if 'benchmark_value' in crisis_data.columns:
            start_benchmark = crisis_data['benchmark_value'].iloc[0]
            end_benchmark = crisis_data['benchmark_value'].iloc[-1]
            result['benchmark_return'] = (end_benchmark / start_benchmark) - 1
            result['outperformance'] = result.get('chronos_return', 0) - result['benchmark_return']
        
        
        if 'drawdown' in crisis_data.columns:
            result['chronos_max_dd'] = crisis_data['drawdown'].min()
        
        
        if 'benchmark_value' in crisis_data.columns:
            cummax = crisis_data['benchmark_value'].cummax()
            benchmark_dd = (crisis_data['benchmark_value'] - cummax) / cummax
            result['benchmark_max_dd'] = benchmark_dd.min()
            result['dd_protection'] = result.get('benchmark_max_dd', 0) - result.get('chronos_max_dd', 0)
        
        
        if 'regime' in crisis_data.columns:
            regime_counts = crisis_data['regime'].value_counts()
            result['dominant_regime'] = regime_counts.index[0]
            result['regime_distribution'] = regime_counts.to_dict()
        
        # Volatility during crisis
        if 'weekly_return' in crisis_data.columns:
            result['crisis_volatility'] = crisis_data['weekly_return'].std() * np.sqrt(52)
        
        result['status'] = 'analyzed'
        
        return result
    
    def get_crisis_summary(self) -> Dict:
        
        analysis = self.analyze_all_crises()
        
        analyzed = analysis[analysis['in_test_period'] == True]
        
        summary = {
            'total_crises_defined': len(CRISIS_PERIODS),
            'crises_in_test_period': len(analyzed),
            'crises_outside_period': len(analysis) - len(analyzed)
        }
        
        if len(analyzed) > 0:
            summary['avg_outperformance'] = analyzed['outperformance'].mean()
            summary['avg_dd_protection'] = analyzed.get('dd_protection', pd.Series([0])).mean()
            summary['total_periods_analyzed'] = analyzed['num_periods'].sum()
        
        return summary
    
    def get_regime_during_crises(self) -> pd.DataFrame:
        
        analysis = self.analyze_all_crises()
        analyzed = analysis[analysis['in_test_period'] == True]
        
        if len(analyzed) == 0:
            return pd.DataFrame()
        
        regime_data = []
        for _, row in analyzed.iterrows():
            if 'regime_distribution' in row and row['regime_distribution']:
                for regime, count in row['regime_distribution'].items():
                    regime_data.append({
                        'crisis': row['name'],
                        'regime': regime,
                        'periods': count
                    })
        
        return pd.DataFrame(regime_data)
