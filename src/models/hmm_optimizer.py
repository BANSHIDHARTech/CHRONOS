"""
HMM Optimizer Module

Implements BIC-based state selection for determining optimal number of regimes.
Tests different covariance types and provides visualization utilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_optimal_states(
    observations: np.ndarray,
    min_states: int = 2,
    max_states: int = 5,
    covariance_type: str = 'full',
    n_iter: int = 1000,
    random_state: int = 42,
    n_init: int = 3
) -> Dict[str, Any]:
    """
    Select optimal number of states using BIC criterion.
    
    Args:
        observations: Feature matrix (n_samples, n_features)
        min_states: Minimum number of states to test
        max_states: Maximum number of states to test
        covariance_type: Covariance type for HMM
        n_iter: Maximum EM iterations
        random_state: Random seed
        n_init: Number of random initializations per state count
        
    Returns:
        Dictionary containing:
            - optimal_n: State count with minimum BIC
            - optimal_model: Fitted model with best BIC
            - bic_scores: List of (n_states, bic) tuples
            - all_results: Detailed results for each state count
    """
    logger.info(f"Selecting optimal states from {min_states} to {max_states}")
    
    if isinstance(observations, pd.DataFrame):
        observations = observations.values
    
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    
    results = []
    bic_scores = []
    
    for n_states in range(min_states, max_states + 1):
        best_bic = np.inf
        best_model = None
        best_log_likelihood = None
        
        # Multiple random initializations
        for init in range(n_init):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=random_state + init,
                    tol=1e-4,
                    verbose=False
                )
                
                model.fit(observations)
                
                # Calculate BIC
                log_likelihood = model.score(observations)
                n_samples = observations.shape[0]
                n_features = observations.shape[1]
                
                # Number of free parameters
                n_params = _count_hmm_parameters(n_states, n_features, covariance_type)
                
                bic = -2 * log_likelihood * n_samples + n_params * np.log(n_samples)
                
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_log_likelihood = log_likelihood
                    
            except Exception as e:
                logger.warning(f"Failed for n_states={n_states}, init={init}: {e}")
                continue
        
        if best_model is not None:
            results.append({
                'n_states': n_states,
                'bic': best_bic,
                'log_likelihood': best_log_likelihood,
                'model': best_model,
                'converged': best_model.monitor_.converged
            })
            bic_scores.append((n_states, best_bic))
            
            logger.info(
                f"  n_states={n_states}: BIC={best_bic:.2f}, "
                f"LL={best_log_likelihood:.2f}, "
                f"converged={best_model.monitor_.converged}"
            )
    
    if not results:
        raise ValueError("No models successfully fitted")
    
    # Find optimal model (minimum BIC)
    optimal_result = min(results, key=lambda x: x['bic'])
    
    logger.info(f"\nOptimal number of states: {optimal_result['n_states']} (BIC={optimal_result['bic']:.2f})")
    
    return {
        'optimal_n': optimal_result['n_states'],
        'optimal_model': optimal_result['model'],
        'optimal_bic': optimal_result['bic'],
        'bic_scores': bic_scores,
        'all_results': results
    }


def _count_hmm_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    """
    Count the number of free parameters in HMM.
    
    Args:
        n_states: Number of hidden states
        n_features: Number of observation features
        covariance_type: Type of covariance matrix
        
    Returns:
        Number of free parameters
    """
    # Initial state probabilities (n_states - 1 free params, sum to 1)
    n_params = n_states - 1
    
    # Transition matrix ((n_states - 1) * n_states free params)
    n_params += n_states * (n_states - 1)
    
    # Means (n_states * n_features)
    n_params += n_states * n_features
    
    # Covariance parameters depend on type
    if covariance_type == 'spherical':
        # One variance per state
        n_params += n_states
    elif covariance_type == 'diag':
        # Diagonal covariance per state
        n_params += n_states * n_features
    elif covariance_type == 'full':
        # Full covariance matrix per state
        n_params += n_states * n_features * (n_features + 1) // 2
    elif covariance_type == 'tied':
        # One shared full covariance matrix
        n_params += n_features * (n_features + 1) // 2
    
    return n_params


def plot_bic_curve(
    bic_scores: List[Tuple[int, float]],
    save_path: Optional[str] = None,
    title: str = "BIC Score vs Number of States"
) -> plt.Figure:
    """
    Create line plot of BIC vs number of states.
    
    Args:
        bic_scores: List of (n_states, bic) tuples
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_states = [x[0] for x in bic_scores]
    bics = [x[1] for x in bic_scores]
    
    # Find optimal point
    optimal_idx = np.argmin(bics)
    optimal_n = n_states[optimal_idx]
    optimal_bic = bics[optimal_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot BIC curve
    ax.plot(n_states, bics, 'bo-', linewidth=2, markersize=8, label='BIC Score')
    
    # Mark optimal point
    ax.axvline(x=optimal_n, color='red', linestyle='--', 
               label=f'Optimal: {optimal_n} states')
    ax.scatter([optimal_n], [optimal_bic], color='red', s=150, zorder=5,
               marker='*', label=f'BIC = {optimal_bic:.0f}')
    
    ax.set_xlabel('Number of States', fontsize=12)
    ax.set_ylabel('BIC Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to integers only
    ax.set_xticks(n_states)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"BIC curve saved to {save_path}")
    
    return fig


def compare_covariance_types(
    observations: np.ndarray,
    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Test different covariance types and compare BIC scores.
    
    Args:
        observations: Feature matrix
        n_states: Number of states to use
        n_iter: Maximum EM iterations
        random_state: Random seed
        
    Returns:
        DataFrame with BIC comparison for each covariance type
    """
    logger.info(f"Comparing covariance types for {n_states} states")
    
    if isinstance(observations, pd.DataFrame):
        observations = observations.values
    
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    
    covariance_types = ['spherical', 'diag', 'full', 'tied']
    results = []
    
    for cov_type in covariance_types:
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                random_state=random_state,
                tol=1e-4,
                verbose=False
            )
            
            model.fit(observations)
            
            # Calculate BIC
            log_likelihood = model.score(observations)
            n_samples = observations.shape[0]
            n_features = observations.shape[1]
            n_params = _count_hmm_parameters(n_states, n_features, cov_type)
            
            bic = -2 * log_likelihood * n_samples + n_params * np.log(n_samples)
            
            results.append({
                'covariance_type': cov_type,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'n_params': n_params,
                'converged': model.monitor_.converged
            })
            
            logger.info(f"  {cov_type}: BIC={bic:.2f}, params={n_params}")
            
        except Exception as e:
            logger.warning(f"Failed for {cov_type}: {e}")
            results.append({
                'covariance_type': cov_type,
                'bic': np.nan,
                'log_likelihood': np.nan,
                'n_params': np.nan,
                'converged': False
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('bic')
    
    # Validate that 'full' performs well for financial data
    best_type = df.iloc[0]['covariance_type']
    if best_type != 'full':
        logger.warning(
            f"Best covariance type is '{best_type}', not 'full'. "
            f"Consider investigating data characteristics."
        )
    
    return df


def calculate_aic(
    model: GaussianHMM,
    observations: np.ndarray,
    n_features: int
) -> float:
    """
    Calculate Akaike Information Criterion.
    
    Args:
        model: Fitted HMM
        observations: Feature matrix
        n_features: Number of features
        
    Returns:
        AIC score
    """
    log_likelihood = model.score(observations)
    n_samples = observations.shape[0]
    n_params = _count_hmm_parameters(
        model.n_components, 
        n_features, 
        model.covariance_type
    )
    
    aic = 2 * n_params - 2 * log_likelihood * n_samples
    return aic


def cross_validate_hmm(
    observations: np.ndarray,
    n_states: int = 3,
    n_folds: int = 5,
    covariance_type: str = 'full',
    n_iter: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform time-series cross-validation for HMM.
    
    Uses expanding window approach to respect temporal ordering.
    
    Args:
        observations: Feature matrix
        n_states: Number of states
        n_folds: Number of CV folds
        covariance_type: Covariance type
        n_iter: Maximum EM iterations
        random_state: Random seed
        
    Returns:
        Dictionary with CV results
    """
    logger.info(f"Cross-validating HMM with {n_folds} folds")
    
    if isinstance(observations, pd.DataFrame):
        observations = observations.values
    
    n_samples = len(observations)
    fold_size = n_samples // (n_folds + 1)
    
    scores = []
    
    for fold in range(n_folds):
        # Expanding window: train on increasing data, test on next fold
        train_end = (fold + 1) * fold_size
        test_end = min((fold + 2) * fold_size, n_samples)
        
        X_train = observations[:train_end]
        X_test = observations[train_end:test_end]
        
        if len(X_test) == 0:
            continue
        
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state,
                tol=1e-4,
                verbose=False
            )
            
            model.fit(X_train)
            test_score = model.score(X_test)
            
            scores.append({
                'fold': fold,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_log_likelihood': test_score
            })
            
            logger.info(f"  Fold {fold}: train={len(X_train)}, test={len(X_test)}, LL={test_score:.2f}")
            
        except Exception as e:
            logger.warning(f"Fold {fold} failed: {e}")
    
    if not scores:
        raise ValueError("No successful CV folds")
    
    # Calculate statistics
    test_scores = [s['test_log_likelihood'] for s in scores]
    
    return {
        'mean_log_likelihood': np.mean(test_scores),
        'std_log_likelihood': np.std(test_scores),
        'fold_results': scores,
        'n_successful_folds': len(scores)
    }


# Example usage
if __name__ == '__main__':
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 500
    
    X = np.column_stack([
        np.random.randn(n_samples) * 0.02,
        np.abs(np.random.randn(n_samples)) * 0.01
    ])
    
    # Test state selection
    result = select_optimal_states(X, min_states=2, max_states=5)
    print(f"\nOptimal states: {result['optimal_n']}")
    print(f"BIC scores: {result['bic_scores']}")
    
    # Test covariance comparison
    cov_comparison = compare_covariance_types(X, n_states=3)
    print(f"\nCovariance comparison:")
    print(cov_comparison)
    
    # Test cross-validation
    cv_result = cross_validate_hmm(X, n_states=3, n_folds=3)
    print(f"\nCV mean LL: {cv_result['mean_log_likelihood']:.2f}")
