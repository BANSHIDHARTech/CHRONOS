# CHRONOS Interpretability Module

SHAP-based model interpretation for regime-specialized XGBoost ensemble.

## Overview

This module provides comprehensive explainability for the CHRONOS trading system's ML models. Using SHAP (SHapley Additive exPlanations), we generate both global and local explanations for predictions across all three market regimes.

## Key Components

### SHAPAnalyzer (`shap_analyzer.py`)
Core class for computing and visualizing SHAP values.

**Features:**
- TreeExplainer integration for efficient XGBoost explanation
- Global explanations: summary plots, beeswarm plots, bar charts
- Local explanations: waterfall plots, force plots
- SHAP value caching for performance
- Automatic directory structure creation

### FeatureImportanceAnalyzer (`feature_importance.py`)
Cross-regime feature importance comparison.

**Features:**
- Compare feature importance across Euphoria/Complacency/Capitulation regimes
- Financial theory validation (checks if expected features are top-ranked)
- Importance heatmaps and comparison bar charts
- Regime-specific feature identification
- Feature stability analysis

### run_shap_analysis.py
End-to-end CLI script for complete SHAP analysis.

## Quick Start

```python
from src.models.model_manager import load_ensemble
from src.interpretability import SHAPAnalyzer, FeatureImportanceAnalyzer

# Load trained ensemble
ensemble = load_ensemble('models/')

# Initialize analyzer
analyzer = SHAPAnalyzer(ensemble, output_dir='outputs/shap')

# Generate plots for Euphoria regime
analyzer.generate_summary_plot(regime_idx=0, X=X_train)
analyzer.generate_beeswarm_plot(regime_idx=0, X=X_train)
analyzer.generate_bar_plot(regime_idx=0, X=X_train)

# Explain individual prediction
analyzer.generate_waterfall_plot(regime_idx=0, X=X_train, sample_idx=42)
```

## CLI Usage

```bash
# Run full analysis on all regimes
python -m src.interpretability.run_shap_analysis

# Analyze specific regime with 10 waterfall samples
python -m src.interpretability.run_shap_analysis --regimes 2 --n-samples 10

# Custom output directory
python -m src.interpretability.run_shap_analysis --output-dir outputs/shap_custom
```

## Output Structure

```
outputs/shap/
├── regime_0_euphoria/
│   ├── summary_plot.png
│   ├── beeswarm_plot.png
│   ├── bar_plot.png
│   └── waterfall_samples/
├── regime_1_complacency/
│   └── [same structure]
├── regime_2_capitulation/
│   └── [same structure]
├── cross_regime/
│   ├── importance_heatmap.png
│   ├── importance_comparison.png
│   ├── regime_specific_features.png
│   ├── feature_importance.csv
│   └── importance_report.md
└── shap_values/
    ├── regime_0_train.npy
    └── [saved SHAP value arrays]
```

## Interpreting SHAP Plots

### Summary Plot
- Each dot represents one sample
- X-axis: SHAP value (impact on prediction)
- Color: feature value (red=high, blue=low)
- Features sorted by mean absolute SHAP value

### Waterfall Plot
- Shows how each feature contributes to a single prediction
- Starts from base value (average prediction)
- Each bar shows feature's contribution
- Final value = model's prediction for that sample

### Importance Heatmap
- Rows: features, Columns: regimes
- Color intensity: importance in that regime
- Helps identify regime-specific vs stable features

## Financial Theory Validation

The module validates that SHAP results align with financial domain knowledge:

| Regime | Expected Top Features |
|--------|----------------------|
| Euphoria | rolling_sharpe, momentum_20d, rsi, macd_histogram |
| Complacency | rolling_sharpe, bb_width, vix_percentile |
| Capitulation | vix_level, vix_percentile, rolling_max_dd, parkinson_vol |

Validation passes if ≥60% of expected features appear in actual top 5.

## Dependencies

- `shap>=0.42.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
