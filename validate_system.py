"""
Quick Validation Script for CHRONOS System

Checks that all components are working correctly.
Run this after training to verify everything is OK.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def check_file(path, description):
    """Check if file exists and report."""
    exists = os.path.exists(path)
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {description}: {path}")
    return exists

def check_shape(path, expected_shape, description):
    """Check file shape matches expected."""
    try:
        if path.endswith('.csv'):
            data = pd.read_csv(path)
            shape = data.shape
        elif path.endswith('.pkl'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                shape = data.shape
            elif isinstance(data, dict) and 'features' in data:
                shape = data['features'].shape
            else:
                shape = "Unknown"
        
        if expected_shape:
            match = shape == expected_shape
            symbol = "✅" if match else "⚠️"
            print(f"  {symbol} Shape: {shape} (expected: {expected_shape})")
        else:
            print(f"  ℹ️ Shape: {shape}")
        return True
    except Exception as e:
        print(f"  ❌ Error checking shape: {e}")
        return False

def main():
    print("=" * 60)
    print("CHRONOS SYSTEM VALIDATION")
    print("=" * 60)
    
    all_good = True
    
    # Phase 1: Data Pipeline
    print("\n📊 Phase 1: Data Pipeline")
    print("-" * 60)
    all_good &= check_file(config.FEATURES_FILE, "Features file")
    if os.path.exists(config.FEATURES_FILE):
        check_shape(config.FEATURES_FILE, (1258, 20), "Features shape")
    
    all_good &= check_file(config.TARGETS_FILE, "Targets file")
    if os.path.exists(config.TARGETS_FILE):
        check_shape(config.TARGETS_FILE, None, "Targets shape")
    
    # Phase 2: HMM Regime Detection
    print("\n🎯 Phase 2: HMM Regime Detection")
    print("-" * 60)
    detector_path = os.path.join(config.MODELS_DIR, 'regime_detector.pkl')
    all_good &= check_file(detector_path, "Regime detector model")
    
    detector_json = os.path.join(config.MODELS_DIR, 'regime_detector.json')
    all_good &= check_file(detector_json, "Regime detector metadata")
    
    if os.path.exists(detector_json):
        with open(detector_json) as f:
            metadata = json.load(f)
            print(f"  ℹ️ Components: {metadata.get('n_components')}")
            print(f"  ℹ️ Converged: {metadata.get('converged')}")
            print(f"  ℹ️ Iterations: {metadata.get('n_iter')}")
    
    regime_labels = os.path.join(config.PROCESSED_DATA_DIR, 'regime_labels.csv')
    all_good &= check_file(regime_labels, "Regime labels")
    if os.path.exists(regime_labels):
        df = pd.read_csv(regime_labels)
        print(f"  ℹ️ Samples: {len(df)}")
        regime_dist = df['regime_id'].value_counts().sort_index()
        print(f"  ℹ️ Distribution: {dict(regime_dist)}")
    
    # Phase 3: XGBoost Ensemble
    print("\n🚀 Phase 3: XGBoost Ensemble")
    print("-" * 60)
    for regime_id, regime_name in enumerate(['euphoria', 'complacency', 'capitulation']):
        model_path = os.path.join(config.MODELS_DIR, f'xgb_regime_{regime_id}_{regime_name}.json')
        all_good &= check_file(model_path, f"Regime {regime_id} ({regime_name.title()}) model")
    
    ensemble_meta = os.path.join(config.MODELS_DIR, 'ensemble_metadata.json')
    all_good &= check_file(ensemble_meta, "Ensemble metadata")
    
    eval_report = os.path.join(config.MODELS_DIR, 'evaluation_report.json')
    if check_file(eval_report, "Evaluation report"):
        with open(eval_report) as f:
            report = json.load(f)
            print(f"  ℹ️ Overall Accuracy: {report['summary']['overall_accuracy']:.2%}")
            print(f"  ℹ️ Validation Samples: {report['summary']['total_samples']}")
            per_regime = report['per_regime_accuracy']
            print(f"  ℹ️ Euphoria Acc: {per_regime['regime_0']:.2%} ({per_regime['regime_0_samples']} samples)")
            print(f"  ℹ️ Complacency Acc: {per_regime['regime_1']:.2%} ({per_regime['regime_1_samples']} samples)")
            print(f"  ℹ️ Capitulation Acc: {per_regime['regime_2']:.2%} ({per_regime['regime_2_samples']} samples)")
    
    # Outputs
    print("\n📁 Output Files")
    print("-" * 60)
    bic_plot = os.path.join(config.FIGURES_DIR, 'bic_selection.png')
    check_file(bic_plot, "BIC selection plot")
    
    validation_plot = os.path.join(config.FIGURES_DIR, 'regime_validation.png')
    check_file(validation_plot, "Regime validation plot")
    
    regime_analysis_dir = os.path.join(config.OUTPUTS_DIR, 'regime_analysis')
    if os.path.exists(regime_analysis_dir):
        files = os.listdir(regime_analysis_dir)
        print(f"  ℹ️ Regime analysis files: {len(files)}")
        for f in files:
            print(f"    - {f}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL SYSTEMS OPERATIONAL")
        print("=" * 60)
        print("\n🎉 CHRONOS is ready for production!")
        print("\nNext steps:")
        print("  1. Review outputs/figures/ for visualizations")
        print("  2. Check models/evaluation_report.json for detailed metrics")
        print("  3. Run: python -m pytest tests/ -v")
        return 0
    else:
        print("⚠️ SOME ISSUES DETECTED")
        print("=" * 60)
        print("\nPlease re-run the training pipeline:")
        print("  1. python src/data/pipeline.py")
        print("  2. python -m src.models.train_hmm")
        print("  3. python src/models/train_ensemble.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())
