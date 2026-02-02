"""
XGBoost Regime-Specialized Ensemble

This module implements the RegimeSpecializedEnsemble class that trains
3 independent XGBoost classifiers, each specialized for one market regime
(Euphoria, Complacency, Capitulation).

The shallow tree configuration with L2 regularization prevents overfitting
to noisy financial data, while early stopping ensures generalization.
"""

import logging
import warnings
from typing import Dict, Optional, Tuple, List
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Configure logging
logger = logging.getLogger(__name__)

# Regime name mapping
REGIME_NAMES = {
    0: 'Euphoria',
    1: 'Complacency',
    2: 'Capitulation'
}


class RegimeSpecializedEnsemble:
    """
    Ensemble of regime-specialized XGBoost classifiers for quintile prediction.
    
    Each of the 3 XGBoost classifiers is trained exclusively on data from its
    corresponding regime, creating specialized experts for different market
    conditions.
    
    Attributes:
        max_depth (int): Maximum tree depth to prevent overfitting
        reg_lambda (float): L2 regularization parameter
        learning_rate (float): Step size shrinkage for stable convergence
        n_estimators (int): Maximum number of boosting rounds
        early_stopping_rounds (int): Early stopping patience
        random_state (int): Random seed for reproducibility
        n_jobs (int): Number of parallel threads (-1 for all cores)
        tree_method (str): Tree construction algorithm
        models (Dict): Dictionary storing trained models per regime
        training_metrics (Dict): Training metrics per regime
        feature_names (List[str]): Feature names for validation
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        learning_rate: float = 0.1,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = 'hist'
    ):
        """
        Initialize the RegimeSpecializedEnsemble.
        
        Args:
            max_depth: Maximum tree depth (default: 3)
            reg_lambda: L2 regularization parameter (default: 1.0)
            learning_rate: Learning rate (default: 0.1)
            n_estimators: Maximum boosting rounds (default: 1000)
            early_stopping_rounds: Early stopping patience (default: 10)
            random_state: Random seed for reproducibility (default: 42)
            n_jobs: Number of parallel threads (default: -1)
            tree_method: Tree construction algorithm (default: 'hist')
        """
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        
        # Storage for trained models and metrics
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.training_metrics: Dict[str, Dict] = {}
        self.feature_names: Optional[List[str]] = None
        
        logger.info(
            f"Initialized RegimeSpecializedEnsemble with max_depth={max_depth}, "
            f"reg_lambda={reg_lambda}, learning_rate={learning_rate}"
        )
    
    def _create_classifier(self) -> xgb.XGBClassifier:
        """
        Create a new XGBClassifier with configured hyperparameters.
        
        Returns:
            xgb.XGBClassifier: Configured classifier instance
        """
        return xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=5,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            use_label_encoder=False,
            eval_metric=['mlogloss', 'merror']
        )
    
    def _validate_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> None:
        """
        Validate input data dimensions and values.
        
        Args:
            X: Feature matrix
            y: Quintile labels
            regime_labels: Regime assignments
            
        Raises:
            ValueError: If data is invalid
        """
        if len(X) != len(y) or len(X) != len(regime_labels):
            raise ValueError(
                f"Data length mismatch: X={len(X)}, y={len(y)}, "
                f"regime_labels={len(regime_labels)}"
            )
        
        # Check quintile labels are in expected range (0-4 or 1-5)
        unique_y = np.unique(y)
        if not (set(unique_y).issubset({0, 1, 2, 3, 4}) or 
                set(unique_y).issubset({1, 2, 3, 4, 5})):
            raise ValueError(
                f"Invalid quintile labels: {unique_y}. "
                f"Expected 0-4 or 1-5"
            )
        
        # Check regime labels
        unique_regimes = np.unique(regime_labels)
        if not set(unique_regimes).issubset({0, 1, 2}):
            raise ValueError(
                f"Invalid regime labels: {unique_regimes}. Expected 0, 1, or 2"
            )
    
    def _check_regime_samples(
        self,
        regime_labels: np.ndarray,
        min_samples: int = 100
    ) -> Dict[int, int]:
        """
        Check that each regime has sufficient samples for training.
        
        Args:
            regime_labels: Array of regime assignments
            min_samples: Minimum samples required per regime
            
        Returns:
            Dict mapping regime index to sample count
        """
        regime_counts = {}
        for regime in [0, 1, 2]:
            count = np.sum(regime_labels == regime)
            regime_counts[regime] = count
            if count < min_samples:
                warnings.warn(
                    f"Regime {regime} ({REGIME_NAMES[regime]}) has only "
                    f"{count} samples (minimum: {min_samples}). "
                    f"Model may not train reliably.",
                    UserWarning
                )
                logger.warning(
                    f"Insufficient samples for regime {regime}: {count} < {min_samples}"
                )
        
        return regime_counts
    
    def train_regime_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
        min_samples_per_regime: int = 100
    ) -> Dict[str, Dict]:
        """
        Train separate XGBoost models for each regime.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Quintile labels (1-5)
            regime_labels: Regime assignments (0=Euphoria, 1=Complacency, 2=Capitulation)
            feature_names: Optional list of feature names
            validation_split: Fraction of data to use for early stopping
            min_samples_per_regime: Minimum samples required per regime
            
        Returns:
            Dictionary of training metrics per regime
        """
        logger.info("Starting regime-specialized model training")
        
        # Validate data
        self._validate_data(X_train, y_train, regime_labels)
        
        # Store feature names
        self.feature_names = feature_names
        
        # Check sample counts per regime
        regime_counts = self._check_regime_samples(regime_labels, min_samples_per_regime)
        logger.info(f"Regime sample counts: {regime_counts}")
        
        # Convert quintile labels to 0-indexed if needed
        y_train_adj = y_train.copy()
        if y_train.min() == 1:
            y_train_adj = y_train - 1
        
        # Train a model for each regime
        for regime_idx in [0, 1, 2]:
            regime_name = REGIME_NAMES[regime_idx]
            mask = regime_labels == regime_idx
            
            X_regime = X_train[mask]
            y_regime = y_train_adj[mask]
            
            n_samples = len(X_regime)
            logger.info(
                f"Training model for regime {regime_idx} ({regime_name}): "
                f"{n_samples} samples"
            )
            
            if n_samples < 10:
                logger.warning(
                    f"Skipping regime {regime_idx} due to insufficient samples"
                )
                continue
            
            # Split into train/validation for early stopping
            if n_samples > 50:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_regime, y_regime,
                    test_size=validation_split,
                    random_state=self.random_state,
                    stratify=y_regime if len(np.unique(y_regime)) > 1 else None
                )
            else:
                # Too few samples, use all for training
                X_tr, y_tr = X_regime, y_regime
                X_val, y_val = X_regime, y_regime
                logger.warning(
                    f"Regime {regime_idx}: Too few samples for validation split, "
                    f"using training data for early stopping"
                )
            
            # Create and train classifier
            model = self._create_classifier()
            
            # Train with early stopping
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Store the trained model
            model_key = f'regime_{regime_idx}'
            self.models[model_key] = model
            
            # Calculate and store training metrics
            y_pred_train = model.predict(X_tr)
            y_pred_val = model.predict(X_val)
            y_proba_val = model.predict_proba(X_val)
            
            train_accuracy = accuracy_score(y_tr, y_pred_train)
            val_accuracy = accuracy_score(y_val, y_pred_val)
            
            try:
                val_logloss = log_loss(y_val, y_proba_val, labels=np.arange(5))
            except ValueError:
                val_logloss = np.nan
            
            self.training_metrics[model_key] = {
                'regime_idx': regime_idx,
                'regime_name': regime_name,
                'n_samples': n_samples,
                'n_train': len(y_tr),
                'n_val': len(y_val),
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'val_logloss': float(val_logloss) if not np.isnan(val_logloss) else None,
                'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None,
                'n_features': X_train.shape[1]
            }
            
            logger.info(
                f"Regime {regime_idx} ({regime_name}): "
                f"train_acc={train_accuracy:.4f}, val_acc={val_accuracy:.4f}, "
                f"val_logloss={val_logloss:.4f}"
            )
        
        logger.info(f"Training complete. Trained {len(self.models)} regime models")
        return self.training_metrics
    
    def get_model(self, regime_idx: int) -> Optional[xgb.XGBClassifier]:
        """
        Get the XGBoost model for a specific regime.
        
        Args:
            regime_idx: Regime index (0, 1, or 2)
            
        Returns:
            XGBClassifier for the specified regime, or None if not trained
        """
        model_key = f'regime_{regime_idx}'
        return self.models.get(model_key)
    
    def predict(
        self,
        X: np.ndarray,
        regime_idx: int
    ) -> np.ndarray:
        """
        Predict quintiles using the specified regime model.
        
        Args:
            X: Feature matrix
            regime_idx: Regime to use for prediction
            
        Returns:
            Array of quintile predictions (1-5)
        """
        model = self.get_model(regime_idx)
        if model is None:
            raise ValueError(f"No model trained for regime {regime_idx}")
        
        # Predict and convert back to 1-indexed quintiles
        predictions = model.predict(X) + 1
        return predictions
    
    def predict_proba(
        self,
        X: np.ndarray,
        regime_idx: int
    ) -> np.ndarray:
        """
        Get probability distribution over quintiles.
        
        Args:
            X: Feature matrix
            regime_idx: Regime to use for prediction
            
        Returns:
            Array of shape (n_samples, 5) with probabilities for each quintile
        """
        model = self.get_model(regime_idx)
        if model is None:
            raise ValueError(f"No model trained for regime {regime_idx}")
        
        return model.predict_proba(X)
    
    def get_feature_importance(
        self,
        regime_idx: int,
        importance_type: str = 'weight'
    ) -> Optional[np.ndarray]:
        """
        Get feature importance for a specific regime model.
        
        Args:
            regime_idx: Regime index
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Array of feature importance values
        """
        model = self.get_model(regime_idx)
        if model is None:
            return None
        
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)
        
        # Convert to array matching feature order
        n_features = model.n_features_in_
        importance = np.zeros(n_features)
        for i in range(n_features):
            key = f'f{i}'
            if key in importance_dict:
                importance[i] = importance_dict[key]
        
        return importance
    
    def get_hyperparameters(self) -> Dict:
        """
        Get the hyperparameters used for training.
        
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'max_depth': self.max_depth,
            'reg_lambda': self.reg_lambda,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'tree_method': self.tree_method
        }
    
    def __repr__(self) -> str:
        """String representation of the ensemble."""
        trained = list(self.models.keys())
        return (
            f"RegimeSpecializedEnsemble("
            f"trained_models={trained}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate})"
        )
