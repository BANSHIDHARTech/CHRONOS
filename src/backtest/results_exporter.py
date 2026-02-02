"""
Results Exporter Module

This module exports backtest results to multiple formats for analysis
and visualization, including CSV, JSON, and dashboard-ready data.
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ResultsExporter:
    
    
    def __init__(self, output_dir: Optional[str] = None):
        
        self.output_dir = output_dir or os.path.join(OUTPUTS_DIR, 'backtest')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ResultsExporter initialized with output dir: {self.output_dir}")
    
    def export_all(
        self,
        results_df: pd.DataFrame,
        summary_stats: Dict,
        regime_performance: pd.DataFrame,
        trade_log: pd.DataFrame,
        regime_transitions: pd.DataFrame,
        crisis_analysis: Optional[pd.DataFrame] = None
    ) -> Dict[str, str]:
        
        output_files = {}
        
        output_files['backtest_results'] = self.export_results_csv(results_df)
        
        output_files['summary_statistics'] = self.export_summary_json(summary_stats)
        
        output_files['regime_performance'] = self.export_regime_csv(regime_performance)
        
        output_files['trade_log'] = self.export_trade_log(trade_log)
        
        output_files['transition_statistics'] = self.export_transitions_json(regime_transitions)
        
        output_files['dashboard_data'] = self.export_dashboard_json(results_df, summary_stats)
        
        output_files['performance_attribution'] = self.export_attribution_csv(results_df)
        
        
        if crisis_analysis is not None:
            output_files['crisis_analysis'] = self.export_crisis_csv(crisis_analysis)
        
        logger.info(f"Exported {len(output_files)} files to {self.output_dir}")
        return output_files
    
    def export_results_csv(self, results_df: pd.DataFrame) -> str:
        
        filepath = os.path.join(self.output_dir, 'backtest_results.csv')
        results_df.to_csv(filepath, index=True)
        logger.info(f"Exported backtest results to {filepath}")
        return filepath
    
    def export_summary_json(self, summary_stats: Dict) -> str:
        
        filepath = os.path.join(self.output_dir, 'summary_statistics.json')
        
        
        clean_stats = {}
        for key, value in summary_stats.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_stats[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_stats[key] = value.tolist()
            else:
                clean_stats[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(clean_stats, f, indent=2, default=str)
        
        logger.info(f"Exported summary statistics to {filepath}")
        return filepath
    
    def export_regime_csv(self, regime_performance: pd.DataFrame) -> str:
        
        filepath = os.path.join(self.output_dir, 'regime_performance.csv')
        regime_performance.to_csv(filepath, index=False)
        logger.info(f"Exported regime performance to {filepath}")
        return filepath
    
    def export_trade_log(self, trade_log: pd.DataFrame) -> str:
        """Export trade log to CSV."""
        filepath = os.path.join(self.output_dir, 'trade_log.csv')
        
        if len(trade_log) > 0:
            flat_log = []
            for _, row in trade_log.iterrows():
                flat_row = {
                    'date': row['date'],
                    'regime': row['regime'],
                    'transaction_cost': row['transaction_cost']
                }
                
                for asset, weight in row.get('old_weights', {}).items():
                    flat_row[f'old_{asset.lower()}_weight'] = weight
                
                for asset, weight in row.get('new_weights', {}).items():
                    flat_row[f'new_{asset.lower()}_weight'] = weight
                flat_log.append(flat_row)
            
            pd.DataFrame(flat_log).to_csv(filepath, index=False)
        else:
            pd.DataFrame().to_csv(filepath, index=False)
        
        logger.info(f"Exported trade log to {filepath}")
        return filepath
    
    def export_transitions_json(self, transitions: pd.DataFrame) -> str:
        
        filepath = os.path.join(self.output_dir, 'transition_statistics.json')
        
        stats = {
            'total_transitions': len(transitions),
            'transitions': transitions.to_dict('records') if len(transitions) > 0 else []
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported transition statistics to {filepath}")
        return filepath
    
    def export_dashboard_json(self, results_df: pd.DataFrame, summary_stats: Dict) -> str:
        
        filepath = os.path.join(self.output_dir, 'dashboard_data.json')
        
        
        dates = results_df.index.strftime('%Y-%m-%d').tolist() if hasattr(results_df.index, 'strftime') else results_df.index.astype(str).tolist()
        
        dashboard_data = {
            'dates': dates,
            'portfolio_values': results_df['portfolio_value'].tolist(),
            'benchmark_values': results_df['benchmark_value'].tolist(),
            'regimes': results_df['regime'].tolist(),
            'drawdowns': results_df['drawdown'].tolist(),
            'weights': {
                'spy': results_df['spy_weight'].tolist() if 'spy_weight' in results_df.columns else [],
                'tlt': results_df['tlt_weight'].tolist() if 'tlt_weight' in results_df.columns else [],
                'gld': results_df['gld_weight'].tolist() if 'gld_weight' in results_df.columns else []
            },
            'summary': {
                'total_return': float(summary_stats.get('total_return_chronos', 0)),
                'benchmark_return': float(summary_stats.get('total_return_benchmark', 0)),
                'sharpe_ratio': float(summary_stats.get('sharpe_ratio_chronos', 0)),
                'max_drawdown': float(summary_stats.get('max_dd_chronos', 0)),
                'win_rate': float(summary_stats.get('win_rate', 0))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"Exported dashboard data to {filepath}")
        return filepath
    
    def export_attribution_csv(self, results_df: pd.DataFrame) -> str:
        
        filepath = os.path.join(self.output_dir, 'performance_attribution.csv')
        
        
        attribution = []
        for asset in ['SPY', 'TLT', 'GLD']:
            weight_col = f'{asset.lower()}_weight'
            if weight_col in results_df.columns:
                avg_weight = results_df[weight_col].mean()
                attribution.append({
                    'asset': asset,
                    'average_weight': avg_weight,
                    'weight_range_min': results_df[weight_col].min(),
                    'weight_range_max': results_df[weight_col].max()
                })
        
        pd.DataFrame(attribution).to_csv(filepath, index=False)
        logger.info(f"Exported performance attribution to {filepath}")
        return filepath
    
    def export_crisis_csv(self, crisis_analysis: pd.DataFrame) -> str:
        
        filepath = os.path.join(self.output_dir, 'crisis_analysis.csv')
        crisis_analysis.to_csv(filepath, index=False)
        logger.info(f"Exported crisis analysis to {filepath}")
        return filepath
    
    def generate_crisis_report(self, crisis_analysis: pd.DataFrame, summary_stats: Dict) -> str:
        
        filepath = os.path.join(self.output_dir, 'crisis_report.md')
        
        report = [
            "# CHRONOS Crisis Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary",
            f"\n- **Test Period**: Portfolio performance during crisis periods",
            f"- **Max Drawdown Protection**: {summary_stats.get('dd_protection', 0):.2%}",
            "\n## Crisis Periods Analyzed\n"
        ]
        
        if len(crisis_analysis) > 0:
            for _, row in crisis_analysis.iterrows():
                report.append(f"### {row.get('name', 'Unknown Crisis')}")
                report.append(f"- Period: {row.get('start', 'N/A')} to {row.get('end', 'N/A')}")
                report.append(f"- In test period: {row.get('in_test_period', False)}")
                if row.get('in_test_period', False):
                    report.append(f"- CHRONOS Return: {row.get('chronos_return', 0):.2%}")
                    report.append(f"- Benchmark Return: {row.get('benchmark_return', 0):.2%}")
                    report.append(f"- Outperformance: {row.get('outperformance', 0):.2%}")
                report.append("")
        else:
            report.append("No crisis periods fall within the backtest range.")
        
        report.append("\n---\n*Report generated by CHRONOS Backtesting System*")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated crisis report at {filepath}")
        return filepath
