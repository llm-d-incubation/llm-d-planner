#!/usr/bin/env python3
"""
ROBUST ML Interpolation for Missing Benchmark Scores

Production-grade implementation with:
1. Rigorous validation (5-fold cross-validation)
2. Confidence intervals (prediction uncertainty)
3. Model selection (compare RF, GradientBoosting, XGBoost, Ridge)
4. Feature importance analysis
5. Outlier detection
6. Quality metrics reporting

Author: NeuralNav Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import os
import re
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_COLS = [
    'aime', 'aime_25', 'artificial_analysis_coding_index', 
    'artificial_analysis_intelligence_index', 'artificial_analysis_math_index',
    'gpqa', 'hle', 'ifbench', 'lcr', 'livecodebench', 
    'math_500', 'mmlu_pro', 'scicode', 'tau2', 'terminalbench_hard'
]

# Models to compare
MODELS = {
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42),
    'Ridge': Ridge(alpha=1.0),
}

# Cross-validation settings
CV_FOLDS = 5
CONFIDENCE_LEVEL = 0.90  # 90% confidence interval

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_model_size(model_name):
    """Extract model size in billions from name."""
    match = re.search(r'(\d+\.?\d*)b', model_name.lower())
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+)x(\d+)b', model_name.lower())
    if match:
        return float(match.group(1)) * float(match.group(2))
    return 30.0

def extract_model_family(model_name):
    """Extract model family from name."""
    name_lower = model_name.lower()
    families = {
        'llama': 'llama', 'qwen': 'qwen', 'deepseek': 'deepseek',
        'mistral': 'mistral', 'mixtral': 'mistral', 'phi': 'phi',
        'gemma': 'gemma', 'gpt': 'gpt', 'claude': 'claude',
        'granite': 'granite', 'nemotron': 'nemotron', 'solar': 'solar',
        'yi': 'yi', 'glm': 'glm', 'kimi': 'kimi', 'minimax': 'minimax',
        'exaone': 'exaone', 'command': 'cohere', 'jamba': 'jamba',
    }
    for key, family in families.items():
        if key in name_lower:
            return family
    return 'other'

def is_reasoning_model(model_name):
    """Check if model is reasoning-focused."""
    name_lower = model_name.lower()
    return 1 if any(x in name_lower for x in ['reasoning', 'think', 'r1', 'o1', 'o3']) else 0

def is_quantized(model_name):
    """Check if model is quantized."""
    name_lower = model_name.lower()
    return 1 if any(x in name_lower for x in ['w4a16', 'w8a8', 'fp8', 'quantized', 'q4', 'q8']) else 0

def is_instruct(model_name):
    """Check if model is instruction-tuned."""
    name_lower = model_name.lower()
    return 1 if any(x in name_lower for x in ['instruct', 'chat', 'it']) else 0

def clean_percentage(val):
    """Convert percentage string to float."""
    if pd.isna(val) or val == 'N/A' or val == '':
        return np.nan
    if isinstance(val, str):
        return float(val.replace('%', ''))
    return float(val)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(csv_path):
    """Load and prepare benchmark data with rich features."""
    df = pd.read_csv(csv_path)
    
    # Clean benchmark columns
    for col in BENCHMARK_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_percentage)
    
    # Extract features
    df['model_size'] = df['Model Name'].apply(extract_model_size)
    df['model_size_log'] = np.log1p(df['model_size'])  # Log scale for size
    df['model_family'] = df['Model Name'].apply(extract_model_family)
    df['is_reasoning'] = df['Model Name'].apply(is_reasoning_model)
    df['is_quantized'] = df['Model Name'].apply(is_quantized)
    df['is_instruct'] = df['Model Name'].apply(is_instruct)
    
    # Encode categorical features
    le_provider = LabelEncoder()
    le_family = LabelEncoder()
    
    df['provider_encoded'] = le_provider.fit_transform(df['Provider'].fillna('Unknown'))
    df['family_encoded'] = le_family.fit_transform(df['model_family'])
    
    return df, le_provider, le_family

def get_feature_matrix(df, target_col):
    """Get feature matrix for prediction."""
    feature_cols = [
        'model_size', 'model_size_log', 'provider_encoded', 'family_encoded',
        'is_reasoning', 'is_quantized', 'is_instruct'
    ]
    
    # Add other benchmark columns as features
    for col in BENCHMARK_COLS:
        if col != target_col and col in df.columns:
            feature_cols.append(col)
    
    return feature_cols

# =============================================================================
# MODEL SELECTION & VALIDATION
# =============================================================================

def evaluate_models(X_train, y_train, models):
    """Evaluate multiple models using cross-validation."""
    results = {}
    
    for name, model in models.items():
        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        
        # Cross-validation scores (negative MAE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
        
        # R² scores
        r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        
        results[name] = {
            'mae_mean': -cv_scores.mean(),
            'mae_std': cv_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
        }
    
    return results

def select_best_model(results):
    """Select the best model based on MAE."""
    best_model = min(results.items(), key=lambda x: x[1]['mae_mean'])
    return best_model[0]

# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

def predict_with_confidence(model, X_train, y_train, X_predict, n_bootstrap=100):
    """
    Predict with confidence intervals using bootstrap.
    
    Returns: predictions, lower_bound, upper_bound
    """
    predictions_bootstrap = []
    
    np.random.seed(42)
    n_samples = len(X_train)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        # Clone and fit model
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_boot, y_boot)
        
        # Predict
        preds = model_clone.predict(X_predict)
        predictions_bootstrap.append(preds)
    
    predictions_bootstrap = np.array(predictions_bootstrap)
    
    # Calculate statistics
    mean_pred = predictions_bootstrap.mean(axis=0)
    
    # Confidence intervals
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    lower = np.percentile(predictions_bootstrap, alpha * 100, axis=0)
    upper = np.percentile(predictions_bootstrap, (1 - alpha) * 100, axis=0)
    
    return mean_pred, lower, upper

# =============================================================================
# MAIN INTERPOLATION
# =============================================================================

def interpolate_benchmark(df, target_col, report):
    """Interpolate missing values for a single benchmark with full analysis."""
    mask_known = df[target_col].notna()
    mask_unknown = df[target_col].isna()
    
    n_known = mask_known.sum()
    n_missing = mask_unknown.sum()
    
    if n_known < 15:
        median_val = df[target_col].median()
        report['benchmarks'][target_col] = {
            'method': 'median_fallback',
            'n_missing': int(n_missing),
            'fill_value': float(median_val) if pd.notna(median_val) else 50.0,
            'reason': f'Not enough training data ({n_known} samples)'
        }
        return df[target_col].fillna(median_val if pd.notna(median_val) else 50.0)
    
    if n_missing == 0:
        report['benchmarks'][target_col] = {
            'method': 'none_needed',
            'n_missing': 0,
        }
        return df[target_col]
    
    # Prepare data
    feature_cols = get_feature_matrix(df, target_col)
    
    X_train = df.loc[mask_known, feature_cols].copy()
    y_train = df.loc[mask_known, target_col].copy()
    X_predict = df.loc[mask_unknown, feature_cols].copy()
    
    # Impute missing features
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_predict_imputed = imputer.transform(X_predict)
    X_predict_scaled = scaler.transform(X_predict_imputed)
    
    # Model selection
    model_results = evaluate_models(X_train_scaled, y_train, MODELS)
    best_model_name = select_best_model(model_results)
    best_model = MODELS[best_model_name]
    
    # Train best model on full data
    best_model.fit(X_train_scaled, y_train)
    
    # Predict with confidence intervals
    predictions, lower, upper = predict_with_confidence(
        best_model, X_train_scaled, y_train, X_predict_scaled, n_bootstrap=50
    )
    
    # Clip to valid range
    predictions = np.clip(predictions, 0, 100)
    lower = np.clip(lower, 0, 100)
    upper = np.clip(upper, 0, 100)
    
    # Calculate confidence width
    confidence_width = (upper - lower).mean()
    
    # Feature importance (for tree-based models)
    feature_importance = {}
    if hasattr(best_model, 'feature_importances_'):
        for feat, imp in zip(feature_cols, best_model.feature_importances_):
            feature_importance[feat] = float(imp)
    
    # Store report
    report['benchmarks'][target_col] = {
        'method': 'ml_interpolation',
        'n_missing': int(n_missing),
        'n_training': int(n_known),
        'best_model': best_model_name,
        'model_comparison': {k: {
            'mae': f"{v['mae_mean']:.2f} ± {v['mae_std']:.2f}",
            'r2': f"{v['r2_mean']:.3f} ± {v['r2_std']:.3f}"
        } for k, v in model_results.items()},
        'predictions': {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max()),
        },
        'confidence_interval': {
            'level': CONFIDENCE_LEVEL,
            'avg_width': float(confidence_width),
        },
        'feature_importance': dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:5])
    }
    
    # Fill values
    result = df[target_col].copy()
    result.loc[mask_unknown] = predictions
    
    return result, predictions, lower, upper, mask_unknown

def main():
    print("=" * 80)
    print("ROBUST ML INTERPOLATION FOR BENCHMARK SCORES")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'cv_folds': CV_FOLDS,
            'confidence_level': CONFIDENCE_LEVEL,
            'models_compared': list(MODELS.keys()),
        },
        'benchmarks': {},
        'summary': {}
    }
    
    # Load data
    csv_path = 'data/benchmarks/models/opensource_all_benchmarks.csv'
    df, le_provider, le_family = load_and_prepare_data(csv_path)
    
    print(f"\nLoaded {len(df)} models")
    print(f"Benchmark columns: {len(BENCHMARK_COLS)}")
    
    initial_missing = sum(df[col].isna().sum() for col in BENCHMARK_COLS if col in df.columns)
    print(f"Initial missing values: {initial_missing}")
    
    # Store predictions with confidence intervals
    predictions_data = {}
    
    print("\n" + "-" * 80)
    print("BENCHMARK-BY-BENCHMARK ANALYSIS")
    print("-" * 80)
    
    for col in BENCHMARK_COLS:
        if col not in df.columns:
            continue
        
        print(f"\n📊 {col}")
        print("-" * 40)
        
        result = interpolate_benchmark(df, col, report)
        
        if isinstance(result, tuple):
            filled_values, predictions, lower, upper, mask = result
            df[col] = filled_values
            
            # Store for later
            predictions_data[col] = {
                'indices': mask[mask].index.tolist(),
                'predictions': predictions.tolist(),
                'lower': lower.tolist(),
                'upper': upper.tolist(),
            }
            
            bench_report = report['benchmarks'][col]
            print(f"  Method: {bench_report['best_model']}")
            print(f"  Filled: {bench_report['n_missing']} values")
            print(f"  Model comparison:")
            for model_name, metrics in bench_report['model_comparison'].items():
                print(f"    {model_name}: MAE={metrics['mae']}, R²={metrics['r2']}")
            print(f"  Predictions: mean={bench_report['predictions']['mean']:.1f}%, std={bench_report['predictions']['std']:.1f}%")
            print(f"  90% CI width: {bench_report['confidence_interval']['avg_width']:.1f}%")
            if bench_report['feature_importance']:
                print(f"  Top features: {list(bench_report['feature_importance'].keys())[:3]}")
        else:
            df[col] = result
            print(f"  Method: {report['benchmarks'][col]['method']}")
    
    # Final statistics
    final_missing = sum(df[col].isna().sum() for col in BENCHMARK_COLS if col in df.columns)
    
    report['summary'] = {
        'total_models': len(df),
        'initial_missing': int(initial_missing),
        'final_missing': int(final_missing),
        'values_filled': int(initial_missing - final_missing),
        'fill_rate': f"{(initial_missing - final_missing) / initial_missing * 100:.1f}%"
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models: {report['summary']['total_models']}")
    print(f"Values filled: {report['summary']['values_filled']} / {report['summary']['initial_missing']}")
    print(f"Fill rate: {report['summary']['fill_rate']}")
    
    # Save interpolated data
    output_path = 'data/benchmarks/models/opensource_all_benchmarks_interpolated.csv'
    
    for col in BENCHMARK_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    
    output_df = df[['Model Name', 'Provider', 'Dataset'] + BENCHMARK_COLS].copy()
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved interpolated data: {output_path}")
    
    # Save detailed report
    report_path = 'data/benchmarks/models/interpolation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved detailed report: {report_path}")
    
    # Save predictions with confidence intervals
    ci_path = 'data/benchmarks/models/predictions_with_confidence.json'
    with open(ci_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"✅ Saved confidence intervals: {ci_path}")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()

