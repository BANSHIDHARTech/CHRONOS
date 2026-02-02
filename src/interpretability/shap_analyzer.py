"""
SHAP Analyzer Module

Provides comprehensive SHAP-based explanations for regime-specialized XGBoost models.
Generates both global (summary plots) and local (waterfall plots) explanations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt
import shap

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regime configuration
REGIME_NAMES = {
    0: 'Euphoria',
    1: 'Complacency',
    2: 'Capitulation'
}

REGIME_COLORS = {
    0: '#2ecc71',  # Green for Euphoria
    1: '#f1c40f',  # Yellow for Complacency
    2: '#e74c3c'   # Red for Capitulation
}


def create_shap_directories(base_dir: str = 'outputs/shap') -> Dict[str, str]:
    """
    Create organized directory structure for SHAP outputs.
    
    Args:
        base_dir: Base directory for SHAP outputs
        
    Returns:
        Dictionary mapping directory names to paths
    """
    directories = {
        'base': base_dir,
        'regime_0': os.path.join(base_dir, 'regime_0_euphoria'),
        'regime_1': os.path.join(base_dir, 'regime_1_complacency'),
        'regime_2': os.path.join(base_dir, 'regime_2_capitulation'),
        'cross_regime': os.path.join(base_dir, 'cross_regime'),
        'shap_values': os.path.join(base_dir, 'shap_values'),
    }
    
    # Add waterfall subdirectories
    for regime_idx in range(3):
        key = f'regime_{regime_idx}_waterfall'
        directories[key] = os.path.join(
            directories[f'regime_{regime_idx}'],
            'waterfall_samples'
        )
    
    # Create all directories
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")
    
    logger.info(f"Created SHAP directory structure at {base_dir}")
    return directories


class SHAPAnalyzer:
    """
    SHAP-based model interpretation for regime-specialized XGBoost ensemble.
    
    Generates global explanations (summary plots, beeswarm, bar charts) and
    local explanations (waterfall plots, force plots) for each regime model.
    
    Attributes:
        ensemble: Trained RegimeSpecializedEnsemble with XGBoost models
        feature_names: List of feature names used by models
        explainers: Dictionary mapping regime indices to TreeExplainers
        shap_values_cache: Cached SHAP values to avoid recomputation
    
    Example:
        >>> from src.models.model_manager import load_ensemble
        >>> ensemble = load_ensemble('models/')
        >>> analyzer = SHAPAnalyzer(ensemble)
        >>> analyzer.generate_summary_plot(regime_idx=0, X=X_train, output_path='outputs/shap/regime_0/summary.png')
    """
    
    def __init__(
        self,
        ensemble: Any,
        feature_names: Optional[List[str]] = None,
        output_dir: str = 'outputs/shap'
    ):
        """
        Initialize SHAP analyzer with trained ensemble.
        
        Args:
            ensemble: Trained RegimeSpecializedEnsemble or dict with 'models' key
            feature_names: List of feature names (if None, extracted from ensemble)
            output_dir: Base directory for SHAP outputs
        """
        self.ensemble = ensemble
        self.output_dir = output_dir
        
        # Extract feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(ensemble, 'feature_names') and ensemble.feature_names:
            self.feature_names = ensemble.feature_names
        elif isinstance(ensemble, dict) and 'feature_names' in ensemble:
            self.feature_names = ensemble['feature_names']
        else:
            self.feature_names = None
            logger.warning("No feature names provided. Plots will use generic names.")
        
        # Initialize explainers dictionary
        self.explainers: Dict[int, shap.TreeExplainer] = {}
        self.shap_values_cache: Dict[str, np.ndarray] = {}
        
        # Create output directories
        self.directories = create_shap_directories(output_dir)
        
        logger.info(f"SHAPAnalyzer initialized with {len(self.feature_names) if self.feature_names else 'unknown'} features")
    
    def _get_model(self, regime_idx: int) -> Any:
        """Get XGBoost model for specified regime."""
        if hasattr(self.ensemble, 'get_model'):
            return self.ensemble.get_model(regime_idx)
        elif hasattr(self.ensemble, 'models'):
            key = f'regime_{regime_idx}'
            return self.ensemble.models.get(key)
        elif isinstance(self.ensemble, dict) and 'models' in self.ensemble:
            return self.ensemble['models'].get(regime_idx)
        else:
            raise ValueError(f"Cannot extract model for regime {regime_idx}")
    
    def _get_or_create_explainer(
        self,
        regime_idx: int,
        background_data: Optional[np.ndarray] = None
    ) -> shap.TreeExplainer:
        """
        Get or create TreeExplainer for specified regime.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            background_data: Optional background data for SHAP computation
            
        Returns:
            SHAP TreeExplainer instance
        """
        if regime_idx in self.explainers:
            return self.explainers[regime_idx]
        
        model = self._get_model(regime_idx)
        if model is None:
            raise ValueError(f"No model found for regime {regime_idx}")
        
        # Create TreeExplainer
        try:
            if background_data is not None:
                # Use subset for efficiency
                bg_sample = background_data[:min(100, len(background_data))]
                explainer = shap.TreeExplainer(model, data=bg_sample)
            else:
                explainer = shap.TreeExplainer(model)
            
            self.explainers[regime_idx] = explainer
            logger.info(f"Created TreeExplainer for regime {regime_idx} ({REGIME_NAMES[regime_idx]})")
            return explainer
            
        except Exception as e:
            logger.error(f"Failed to create TreeExplainer for regime {regime_idx}: {e}")
            raise
    
    def compute_shap_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regime_idx: int,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for given data and regime.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            regime_idx: Regime index (0, 1, or 2)
            cache_key: Optional key for caching results
            
        Returns:
            SHAP values matrix (n_samples, n_features)
        """
        # Check cache
        if cache_key and cache_key in self.shap_values_cache:
            logger.info(f"Using cached SHAP values for {cache_key}")
            return self.shap_values_cache[cache_key]
        
        # Convert DataFrame to numpy
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(X.columns)
            X = X.values
        
        # Get explainer
        explainer = self._get_or_create_explainer(regime_idx, background_data=X)
        
        # Compute SHAP values
        logger.info(f"Computing SHAP values for regime {regime_idx} on {X.shape[0]} samples...")
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class output from XGBoost
        # SHAP can return: list of arrays, 3D array, or 2D array
        if isinstance(shap_values, list):
            # Old SHAP format: list of arrays per class
            # Average across classes -> shape becomes (samples, features)
            shap_values = np.mean(np.stack(shap_values, axis=-1), axis=-1)
        elif len(shap_values.shape) == 3:
            # New SHAP format: 3D array (samples, features, classes)
            # Average across classes -> shape becomes (samples, features)
            shap_values = np.mean(shap_values, axis=2)
        # If 2D, it's already in correct format
        
        # Cache results
        if cache_key:
            self.shap_values_cache[cache_key] = shap_values
        
        logger.info(f"Computed SHAP values shape: {shap_values.shape}")
        return shap_values
    
    def compute_all_regimes(
        self,
        X_dict: Dict[int, Union[np.ndarray, pd.DataFrame]]
    ) -> Dict[int, np.ndarray]:
        """
        Compute SHAP values for all regimes.
        
        Args:
            X_dict: Dictionary mapping regime indices to feature matrices
            
        Returns:
            Dictionary mapping regime indices to SHAP values
        """
        results = {}
        for regime_idx, X in X_dict.items():
            cache_key = f"regime_{regime_idx}"
            results[regime_idx] = self.compute_shap_values(X, regime_idx, cache_key)
        return results
    
    # =========================================================================
    # Global Explanation Methods
    # =========================================================================
    
    def generate_summary_plot(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        output_path: Optional[str] = None,
        max_display: int = 15
    ) -> str:
        """
        Generate SHAP summary plot showing feature importance.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            output_path: Path to save plot (auto-generated if None)
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(
                self.directories[f'regime_{regime_idx}'],
                'summary_plot.png'
            )
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X, regime_idx)
        
        # Convert X to numpy if needed
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Create plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_np,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Feature Importance - {REGIME_NAMES[regime_idx]} Regime", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary plot to {output_path}")
        return output_path
    
    def generate_beeswarm_plot(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        output_path: Optional[str] = None,
        max_display: int = 15
    ) -> str:
        """
        Generate SHAP beeswarm plot showing feature impact distribution.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            output_path: Path to save plot
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(
                self.directories[f'regime_{regime_idx}'],
                'beeswarm_plot.png'
            )
        
        shap_values = self.compute_shap_values(X, regime_idx)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_np,
            feature_names=self.feature_names,
            plot_type='dot',
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Beeswarm - {REGIME_NAMES[regime_idx]} Regime", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved beeswarm plot to {output_path}")
        return output_path
    
    def generate_bar_plot(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        output_path: Optional[str] = None,
        max_display: int = 15
    ) -> str:
        """
        Generate bar chart of mean absolute SHAP values.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            output_path: Path to save plot
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(
                self.directories[f'regime_{regime_idx}'],
                'bar_plot.png'
            )
        
        shap_values = self.compute_shap_values(X, regime_idx)
        
        # Compute mean absolute SHAP values
        # SHAP values should be 2D now (samples, features) after processing
        # Take mean across samples to get per-feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure mean_shap is 1D
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=-1)  # Average any remaining dimensions
        
        # Sort by importance
        feature_names = self.feature_names or [f'Feature {i}' for i in range(len(mean_shap))]
        sorted_idx = np.argsort(mean_shap)[::-1][:max_display]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        colors = [REGIME_COLORS[regime_idx]] * len(sorted_idx)
        plt.barh(
            range(len(sorted_idx)),
            mean_shap[sorted_idx][::-1],
            color=colors,
            alpha=0.8
        )
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx[::-1]])
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.title(f"Feature Importance - {REGIME_NAMES[regime_idx]} Regime", fontsize=14)
        
        # Add value labels
        for i, (idx, val) in enumerate(zip(sorted_idx[::-1], mean_shap[sorted_idx][::-1])):
            plt.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved bar plot to {output_path}")
        return output_path
    
    # =========================================================================
    # Local Explanation Methods
    # =========================================================================
    
    def generate_waterfall_plot(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        sample_idx: int,
        output_path: Optional[str] = None,
        sample_metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate waterfall plot explaining single prediction.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            sample_idx: Index of sample to explain
            output_path: Path to save plot
            sample_metadata: Optional dict with date, quintile, confidence
            
        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(
                self.directories[f'regime_{regime_idx}_waterfall'],
                f'sample_{sample_idx:03d}.png'
            )
        
        shap_values = self.compute_shap_values(X, regime_idx)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        explainer = self._get_or_create_explainer(regime_idx)
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value.mean()
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected_value,
            data=X_np[sample_idx],
            feature_names=self.feature_names
        )
        
        # Create plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False, max_display=12)
        
        # Add title with metadata
        title = f"Prediction Explanation - {REGIME_NAMES[regime_idx]} Regime"
        if sample_metadata:
            date = sample_metadata.get('date', '')
            quintile = sample_metadata.get('quintile', '')
            confidence = sample_metadata.get('confidence', '')
            title += f"\n{date} | Quintile: {quintile} | Confidence: {confidence:.2f}" if confidence else ''
        plt.title(title, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved waterfall plot to {output_path}")
        return output_path
    
    def generate_waterfall_batch(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        sample_indices: List[int],
        sample_metadata_list: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Generate waterfall plots for multiple samples.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            sample_indices: List of sample indices to explain
            sample_metadata_list: Optional list of metadata dicts
            
        Returns:
            List of paths to saved plots
        """
        paths = []
        for i, sample_idx in enumerate(sample_indices):
            metadata = sample_metadata_list[i] if sample_metadata_list else None
            path = self.generate_waterfall_plot(regime_idx, X, sample_idx, sample_metadata=metadata)
            paths.append(path)
        
        logger.info(f"Generated {len(paths)} waterfall plots for regime {regime_idx}")
        return paths
    
    def generate_force_plot(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        sample_idx: int,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate force plot for single prediction.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            sample_idx: Index of sample to explain
            output_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = os.path.join(
                self.directories[f'regime_{regime_idx}_waterfall'],
                f'force_{sample_idx:03d}.html'
            )
        
        shap_values = self.compute_shap_values(X, regime_idx)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        explainer = self._get_or_create_explainer(regime_idx)
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value.mean()
        
        # Generate force plot HTML
        force_plot = shap.force_plot(
            expected_value,
            shap_values[sample_idx],
            X_np[sample_idx],
            feature_names=self.feature_names,
            matplotlib=False
        )
        shap.save_html(output_path, force_plot)
        
        logger.info(f"Saved force plot to {output_path}")
        return output_path
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_feature_importance_dict(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Get feature importance as dictionary.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            
        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        shap_values = self.compute_shap_values(X, regime_idx)
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure mean_shap is 1D
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=-1)
        
        feature_names = self.feature_names or [f'Feature {i}' for i in range(len(mean_shap))]
        return dict(zip(feature_names, mean_shap))
    
    def save_shap_values(
        self,
        regime_idx: int,
        shap_values: np.ndarray,
        split_name: str = 'train'
    ) -> str:
        """
        Save SHAP values to disk as numpy array.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            shap_values: SHAP values array
            split_name: Data split name (train, val, test)
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(
            self.directories['shap_values'],
            f'regime_{regime_idx}_{split_name}.npy'
        )
        np.save(output_path, shap_values)
        logger.info(f"Saved SHAP values to {output_path}")
        return output_path
    
    def load_shap_values(self, input_path: str) -> np.ndarray:
        """
        Load previously computed SHAP values.
        
        Args:
            input_path: Path to .npy file
            
        Returns:
            SHAP values array
        """
        shap_values = np.load(input_path)
        logger.info(f"Loaded SHAP values from {input_path}, shape: {shap_values.shape}")
        return shap_values
    
    def select_interesting_samples(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        confidences: np.ndarray,
        n_samples: int = 5
    ) -> List[int]:
        """
        Select interesting samples for waterfall plots.
        
        Selects samples with:
        - High confidence (>0.8)
        - Low confidence (<0.4)
        - Median confidence
        
        Args:
            regime_idx: Regime index
            X: Feature matrix
            confidences: Prediction confidence array
            n_samples: Number of samples to select per category
            
        Returns:
            List of sample indices
        """
        indices = []
        
        # High confidence samples
        high_conf_mask = confidences > 0.8
        if high_conf_mask.any():
            high_conf_idx = np.where(high_conf_mask)[0][:n_samples]
            indices.extend(high_conf_idx.tolist())
        
        # Low confidence samples
        low_conf_mask = confidences < 0.4
        if low_conf_mask.any():
            low_conf_idx = np.where(low_conf_mask)[0][:n_samples]
            indices.extend(low_conf_idx.tolist())
        
        # Median confidence samples
        median_conf = np.median(confidences)
        median_mask = np.abs(confidences - median_conf) < 0.05
        if median_mask.any():
            median_idx = np.where(median_mask)[0][:n_samples]
            indices.extend(median_idx.tolist())
        
        return list(set(indices))  # Remove duplicates
    
    def generate_all_plots(
        self,
        regime_idx: int,
        X: Union[np.ndarray, pd.DataFrame],
        n_waterfall_samples: int = 5,
        confidences: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Generate all SHAP plots for a regime.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            X: Feature matrix
            n_waterfall_samples: Number of waterfall plots to generate
            confidences: Optional prediction confidences for sample selection
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        paths = {}
        
        # Global plots
        paths['summary'] = self.generate_summary_plot(regime_idx, X)
        paths['beeswarm'] = self.generate_beeswarm_plot(regime_idx, X)
        paths['bar'] = self.generate_bar_plot(regime_idx, X)
        
        # Waterfall plots
        if confidences is not None:
            sample_indices = self.select_interesting_samples(
                regime_idx, X, confidences, n_waterfall_samples
            )
        else:
            sample_indices = list(range(min(n_waterfall_samples, len(X))))
        
        paths['waterfall'] = self.generate_waterfall_batch(regime_idx, X, sample_indices)
        
        # Save SHAP values
        shap_values = self.compute_shap_values(X, regime_idx)
        paths['shap_values'] = self.save_shap_values(regime_idx, shap_values)
        
        logger.info(f"Generated all plots for regime {regime_idx}")
        return paths
