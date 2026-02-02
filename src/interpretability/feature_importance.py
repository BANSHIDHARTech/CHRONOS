"""
Feature Importance Analyzer Module

Cross-regime feature analysis using SHAP values to compare feature importance
across Euphoria, Complacency, and Capitulation regimes.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .shap_analyzer import SHAPAnalyzer, REGIME_NAMES, REGIME_COLORS

logger = logging.getLogger(__name__)

# Expected top features per regime based on financial theory
EXPECTED_TOP_FEATURES = {
    0: ['rolling_sharpe', 'momentum_20d', 'rsi', 'macd_histogram', 'log_returns'],  # Euphoria
    1: ['rolling_sharpe', 'bb_width', 'vix_percentile', 'rsi', 'parkinson_vol'],    # Complacency
    2: ['vix_level', 'vix_percentile', 'rolling_max_dd', 'parkinson_vol', 'atr']    # Capitulation
}


class FeatureImportanceAnalyzer:
    """
    Cross-regime feature importance analysis using SHAP values.
    
    Compares feature importance across market regimes, validates alignment
    with financial theory, and generates comparison visualizations.
    
    Attributes:
        shap_analyzer: SHAPAnalyzer instance for computing SHAP values
        importance_df: DataFrame with cross-regime feature importance
        
    Example:
        >>> from src.interpretability.shap_analyzer import SHAPAnalyzer
        >>> shap_analyzer = SHAPAnalyzer(ensemble)
        >>> fi_analyzer = FeatureImportanceAnalyzer(shap_analyzer)
        >>> fi_analyzer.compute_cross_regime_importance(X_train, X_val, X_test, regime_labels)
        >>> fi_analyzer.plot_importance_heatmap('outputs/shap/cross_regime/heatmap.png')
    """
    
    def __init__(
        self,
        shap_analyzer: SHAPAnalyzer,
        output_dir: str = 'outputs/shap/cross_regime'
    ):
        """
        Initialize feature importance analyzer.
        
        Args:
            shap_analyzer: Initialized SHAPAnalyzer instance
            output_dir: Directory for saving outputs
        """
        self.shap_analyzer = shap_analyzer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.importance_df: Optional[pd.DataFrame] = None
        self.regime_importance: Dict[int, Dict[str, float]] = {}
        
        logger.info("FeatureImportanceAnalyzer initialized")
    
    def compute_cross_regime_importance(
        self,
        X_dict: Dict[int, np.ndarray],
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute feature importance across all regimes.
        
        Args:
            X_dict: Dictionary mapping regime indices to feature matrices
            normalize: Whether to normalize importance to sum to 1.0
            
        Returns:
            DataFrame with cross-regime feature importance
        """
        logger.info("Computing cross-regime feature importance...")
        
        importance_data = {}
        regime_sample_counts = {}
        
        for regime_idx, X in X_dict.items():
            # Get feature importance dict
            importance = self.shap_analyzer.get_feature_importance_dict(regime_idx, X)
            self.regime_importance[regime_idx] = importance
            regime_sample_counts[regime_idx] = len(X)
            
            # Normalize if requested
            if normalize:
                total = sum(importance.values())
                importance = {k: v / total for k, v in importance.items()}
            
            importance_data[f'regime_{regime_idx}_{REGIME_NAMES[regime_idx].lower()}'] = importance
        
        # Create DataFrame
        self.importance_df = pd.DataFrame(importance_data)
        
        # Calculate overall importance (weighted by sample count)
        total_samples = sum(regime_sample_counts.values())
        weights = {r: c / total_samples for r, c in regime_sample_counts.items()}
        
        overall = np.zeros(len(self.importance_df))
        for regime_idx in X_dict.keys():
            col = f'regime_{regime_idx}_{REGIME_NAMES[regime_idx].lower()}'
            overall += self.importance_df[col].values * weights[regime_idx]
        
        self.importance_df['overall_importance'] = overall
        
        # Sort by overall importance
        self.importance_df = self.importance_df.sort_values('overall_importance', ascending=False)
        
        # Add rank column
        self.importance_df['rank'] = range(1, len(self.importance_df) + 1)
        
        logger.info(f"Computed importance for {len(self.importance_df)} features across {len(X_dict)} regimes")
        return self.importance_df
    
    def rank_features_by_regime(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top N features per regime with rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping regime indices to list of (feature, importance) tuples
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        rankings = {}
        for regime_idx in range(3):
            col = f'regime_{regime_idx}_{REGIME_NAMES[regime_idx].lower()}'
            sorted_df = self.importance_df.sort_values(col, ascending=False)
            top_features = [(idx, sorted_df.loc[idx, col]) for idx in sorted_df.head(top_n).index]
            rankings[regime_idx] = top_features
        
        return rankings
    
    def identify_regime_specific_features(self, threshold: float = 1.5) -> Dict[int, List[str]]:
        """
        Find features disproportionately important in one regime.
        
        A feature is regime-specific if its importance in that regime
        is > threshold times the mean importance across regimes.
        
        Args:
            threshold: Ratio threshold for regime-specificity
            
        Returns:
            Dictionary mapping regime indices to list of specific features
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        specific_features = {0: [], 1: [], 2: []}
        
        for feature in self.importance_df.index:
            regime_cols = [f'regime_{i}_{REGIME_NAMES[i].lower()}' for i in range(3)]
            values = self.importance_df.loc[feature, regime_cols].values
            mean_val = values.mean()
            
            if mean_val == 0:
                continue
            
            for i, val in enumerate(values):
                ratio = val / mean_val
                if ratio > threshold:
                    specific_features[i].append(feature)
        
        for regime_idx, features in specific_features.items():
            logger.info(f"Regime {REGIME_NAMES[regime_idx]} specific features: {features}")
        
        return specific_features
    
    def validate_financial_theory(self) -> Dict[str, Any]:
        """
        Check if SHAP results align with financial domain knowledge.
        
        Validates that expected features for each regime are in top 5.
        
        Returns:
            Dictionary with validation results per regime
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        results = {}
        
        for regime_idx in range(3):
            col = f'regime_{regime_idx}_{REGIME_NAMES[regime_idx].lower()}'
            actual_top5 = self.importance_df.sort_values(col, ascending=False).head(5).index.tolist()
            expected = EXPECTED_TOP_FEATURES.get(regime_idx, [])
            
            # Calculate overlap
            overlap = len(set(expected) & set(actual_top5))
            overlap_pct = (overlap / len(expected)) * 100 if expected else 0
            
            results[regime_idx] = {
                'regime_name': REGIME_NAMES[regime_idx],
                'expected_features': expected,
                'actual_top5': actual_top5,
                'overlap_count': overlap,
                'overlap_pct': overlap_pct,
                'validated': overlap_pct >= 60
            }
            
            status = "PASS" if results[regime_idx]['validated'] else "WARN"
            logger.info(f"[{status}] Financial Theory Validation - {REGIME_NAMES[regime_idx]}: "
                       f"{overlap_pct:.1f}% match ({overlap}/{len(expected)} features)")
        
        return results
    
    def compute_feature_stability(self) -> pd.DataFrame:
        """
        Measure how consistent feature importance is across regimes.
        
        Low variance = stable importance across regimes
        High variance = regime-dependent importance
        
        Returns:
            DataFrame with stability metrics per feature
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        regime_cols = [f'regime_{i}_{REGIME_NAMES[i].lower()}' for i in range(3)]
        
        stability_df = pd.DataFrame({
            'feature': self.importance_df.index,
            'mean_importance': self.importance_df[regime_cols].mean(axis=1).values,
            'std_importance': self.importance_df[regime_cols].std(axis=1).values,
            'cv': self.importance_df[regime_cols].std(axis=1).values / 
                  (self.importance_df[regime_cols].mean(axis=1).values + 1e-10),
            'max_regime_diff': self.importance_df[regime_cols].max(axis=1).values - 
                               self.importance_df[regime_cols].min(axis=1).values
        })
        
        stability_df = stability_df.sort_values('cv', ascending=True)
        stability_df['stability_rank'] = range(1, len(stability_df) + 1)
        
        return stability_df
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def plot_importance_heatmap(self, output_path: Optional[str] = None) -> str:
        """
        Generate heatmap showing feature importance across regimes.
        
        Args:
            output_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'importance_heatmap.png')
        
        # Prepare data - take top 15 features
        regime_cols = [f'regime_{i}_{REGIME_NAMES[i].lower()}' for i in range(3)]
        plot_df = self.importance_df[regime_cols].head(15)
        
        # Rename columns for display
        plot_df.columns = [REGIME_NAMES[i] for i in range(3)]
        
        # Create heatmap
        plt.figure(figsize=(10, 12))
        sns.heatmap(
            plot_df,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Mean |SHAP Value|'},
            linewidths=0.5
        )
        plt.title('Feature Importance Across Market Regimes', fontsize=14)
        plt.xlabel('Market Regime', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved importance heatmap to {output_path}")
        return output_path
    
    def plot_importance_comparison_bars(
        self,
        output_path: Optional[str] = None,
        top_n: int = 10
    ) -> str:
        """
        Generate grouped bar chart comparing top features across regimes.
        
        Args:
            output_path: Path to save plot
            top_n: Number of top features to display
            
        Returns:
            Path to saved plot
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'importance_comparison.png')
        
        # Get top N features by overall importance
        top_features = self.importance_df.head(top_n).index.tolist()
        
        # Prepare data
        regime_cols = [f'regime_{i}_{REGIME_NAMES[i].lower()}' for i in range(3)]
        plot_data = self.importance_df.loc[top_features, regime_cols]
        
        # Create grouped bar chart
        x = np.arange(len(top_features))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (col, regime_idx) in enumerate(zip(regime_cols, range(3))):
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                plot_data[col],
                width,
                label=REGIME_NAMES[regime_idx],
                color=REGIME_COLORS[regime_idx],
                alpha=0.8
            )
        
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title('Feature Importance Comparison Across Regimes', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved importance comparison to {output_path}")
        return output_path
    
    def plot_regime_specific_features(
        self,
        output_path: Optional[str] = None,
        threshold: float = 1.5
    ) -> str:
        """
        Highlight features unique to each regime.
        
        Args:
            output_path: Path to save plot
            threshold: Threshold for regime-specificity
            
        Returns:
            Path to saved plot
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'regime_specific_features.png')
        
        specific_features = self.identify_regime_specific_features(threshold)
        
        # Create scatter plot data
        scatter_data = []
        for regime_idx, features in specific_features.items():
            col = f'regime_{regime_idx}_{REGIME_NAMES[regime_idx].lower()}'
            for feature in features:
                importance = self.importance_df.loc[feature, col]
                mean_importance = self.importance_df.loc[feature, 'overall_importance']
                ratio = importance / (mean_importance + 1e-10)
                scatter_data.append({
                    'feature': feature,
                    'regime': REGIME_NAMES[regime_idx],
                    'regime_idx': regime_idx,
                    'importance': importance,
                    'ratio': ratio
                })
        
        if not scatter_data:
            logger.warning("No regime-specific features found")
            # Create empty placeholder
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No regime-specific features found\n(threshold too high?)',
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for regime_idx in scatter_df['regime_idx'].unique():
            regime_data = scatter_df[scatter_df['regime_idx'] == regime_idx]
            plt.scatter(
                regime_data['regime'],
                regime_data['feature'],
                s=regime_data['ratio'] * 100,
                c=REGIME_COLORS[regime_idx],
                alpha=0.7,
                label=REGIME_NAMES[regime_idx]
            )
            
            # Annotate
            for _, row in regime_data.iterrows():
                plt.annotate(
                    f"{row['ratio']:.1f}x",
                    (row['regime'], row['feature']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8
                )
        
        plt.xlabel('Regime', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Regime-Specific Features (>{threshold}x mean importance)', fontsize=14)
        plt.legend(title='Regime')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved regime-specific features plot to {output_path}")
        return output_path
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_importance_table(self, output_path: Optional[str] = None) -> str:
        """
        Export feature importance as CSV.
        
        Args:
            output_path: Path to save CSV
            
        Returns:
            Path to saved file
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'feature_importance.csv')
        
        # Format for export
        export_df = self.importance_df.copy()
        export_df.index.name = 'feature'
        
        # Rename columns for clarity
        rename_map = {
            f'regime_{i}_{REGIME_NAMES[i].lower()}': f'Regime_{i}_{REGIME_NAMES[i]}'
            for i in range(3)
        }
        export_df = export_df.rename(columns=rename_map)
        
        # Round values
        for col in export_df.columns:
            if export_df[col].dtype == float:
                export_df[col] = export_df[col].round(4)
        
        export_df.to_csv(output_path)
        logger.info(f"Exported importance table to {output_path}")
        return output_path
    
    def generate_importance_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate markdown report summarizing key findings.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        if self.importance_df is None:
            raise ValueError("Must call compute_cross_regime_importance first")
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'importance_report.md')
        
        # Get data for report
        rankings = self.rank_features_by_regime(top_n=5)
        validation = self.validate_financial_theory()
        specific = self.identify_regime_specific_features()
        stability = self.compute_feature_stability()
        
        # Build report
        report = []
        report.append("# Feature Importance Analysis Report\n")
        report.append(f"Generated from SHAP analysis of regime-specialized XGBoost ensemble.\n")
        
        # Top features per regime
        report.append("\n## Top 5 Features by Regime\n")
        for regime_idx in range(3):
            report.append(f"\n### {REGIME_NAMES[regime_idx]} (Regime {regime_idx})\n")
            report.append("| Rank | Feature | Importance |")
            report.append("|------|---------|------------|")
            for i, (feature, importance) in enumerate(rankings[regime_idx], 1):
                report.append(f"| {i} | {feature} | {importance:.4f} |")
        
        # Financial theory validation
        report.append("\n## Financial Theory Validation\n")
        for regime_idx, result in validation.items():
            status = "[PASS]" if result['validated'] else "[WARN]"
            report.append(f"\n### {status} {result['regime_name']}")
            report.append(f"\n- **Expected**: {', '.join(result['expected_features'])}")
            report.append(f"- **Actual Top 5**: {', '.join(result['actual_top5'])}")
            report.append(f"- **Overlap**: {result['overlap_pct']:.1f}% ({result['overlap_count']}/5)")
        
        # Regime-specific features
        report.append("\n## Regime-Specific Features\n")
        report.append("Features with importance >1.5x the cross-regime mean:\n")
        for regime_idx, features in specific.items():
            if features:
                report.append(f"\n**{REGIME_NAMES[regime_idx]}**: {', '.join(features)}")
            else:
                report.append(f"\n**{REGIME_NAMES[regime_idx]}**: None identified")
        
        # Most stable features
        report.append("\n## Feature Stability\n")
        report.append("Features with consistent importance across regimes (low CV):\n")
        top_stable = stability.head(5)
        report.append("| Feature | Mean Importance | CV | Stability Rank |")
        report.append("|---------|-----------------|-----|----------------|")
        for _, row in top_stable.iterrows():
            report.append(f"| {row['feature']} | {row['mean_importance']:.4f} | {row['cv']:.2f} | {int(row['stability_rank'])} |")
        
        # Write report
        report_text = '\n'.join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Generated importance report at {output_path}")
        return output_path
    
    def generate_all_outputs(self) -> Dict[str, str]:
        """
        Generate all visualizations and exports.
        
        Returns:
            Dictionary mapping output types to file paths
        """
        outputs = {}
        
        outputs['heatmap'] = self.plot_importance_heatmap()
        outputs['comparison_bars'] = self.plot_importance_comparison_bars()
        outputs['regime_specific'] = self.plot_regime_specific_features()
        outputs['table'] = self.export_importance_table()
        outputs['report'] = self.generate_importance_report()
        
        logger.info(f"Generated {len(outputs)} outputs")
        return outputs
