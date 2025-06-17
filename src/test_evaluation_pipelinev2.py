#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Add src to path for imports
sys.path.append('src')

from data_preprocessor import AdultDataPreprocessor
from synthesizers.privbayes import MyPrivBayes
from synthesizers.dp_ctgan_opacus import OpacusDifferentiallyPrivateCTGAN

def load_and_preprocess_adult_data(synthesizer_type, train_size=1000, test_size=500):
    """
    Load and preprocess the Adult dataset using the unified preprocessing system.
    
    Args:
        synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
        train_size (int): Number of training samples to use
        test_size (int): Number of test samples to use
        
    Returns:
        tuple: (real_train, real_test, preprocessing_metadata)
    """
    print(f"=" * 80)
    print(f"LOADING AND PREPROCESSING DATA FOR {synthesizer_type.upper()}")
    print(f"=" * 80)
    
    # Initialize preprocessor with absolute path to root data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    preprocessor = AdultDataPreprocessor(data_dir=data_dir)
    
    # Download data if not exists
    print("📥 Downloading Adult dataset if needed...")
    preprocessor.download()
    
    # Load data
    print("📋 Loading Adult dataset...")
    df_train, df_test = preprocessor.load()
    
    print("\nData types in original data:")
    print(df_train.dtypes)
    print("\nSample of original data:")
    print(df_train.head())
    
    # Combine and split to ensure consistent preprocessing
    print("\n🔄 Combining and splitting data...")
    df_train_combined, df_test_combined = preprocessor.combine_and_split(
        df_train, df_test, test_size=0.3, random_state=42
    )
    
    # Use unified preprocessing system
    print(f"\n⚙️  Preprocessing data for {synthesizer_type}...")
    df_train_processed, train_metadata = preprocessor.preprocess_for_synthesizer(
        df_train_combined, synthesizer_type
    )
    
    # CRITICAL: Use the same preprocessing for test data to ensure TSTR consistency
    df_test_processed, test_metadata = preprocessor.preprocess_for_synthesizer(
        df_test_combined, synthesizer_type
    )
    
    print("\nData types after preprocessing:")
    print(df_train_processed.dtypes)
    print("\nSample of preprocessed data:")
    print(df_train_processed.head())
    
    # Verify metadata consistency
    assert train_metadata == test_metadata, "Train and test preprocessing metadata must match!"
    
    # Take samples for faster testing
    print("\n📊 Taking samples for faster testing...")
    train_sample = df_train_processed.sample(n=min(train_size, len(df_train_processed)), random_state=42)
    test_sample = df_test_processed.sample(n=min(test_size, len(df_test_processed)), random_state=42)
    
    print(f"\n✅ Training sample shape: {train_sample.shape}")
    print(f"✅ Test sample shape: {test_sample.shape}")
    print(f"📋 Preprocessing metadata: {train_metadata}")
    
    return train_sample, test_sample, train_metadata

def test_synthesizer(synthesizer_type, real_train, preprocessing_metadata, n_synthetic=800, epsilon=1.0):
    """
    Test synthesis for specified synthesizer type.
    
    Args:
        synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
        real_train (pd.DataFrame): Preprocessed training data
        preprocessing_metadata (dict): Metadata from preprocessing
        n_synthetic (int): Number of synthetic samples to generate
        epsilon (float): Privacy budget parameter
        
    Returns:
        pd.DataFrame or None: Generated synthetic data
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING {synthesizer_type.upper()} SYNTHESIS")
    print(f"{'=' * 80}")
    
    print(f"📊 Real training data shape: {real_train.shape}")
    
    # Initialize synthesizer based on type
    if synthesizer_type.lower() == 'privbayes':
        print(f"\n🚀 Initializing PrivBayes synthesizer...")
        synthesizer = MyPrivBayes(epsilon=epsilon)
        
        # Fit the synthesizer
        print("🔧 Fitting PrivBayes on training data...")
        try:
            synthesizer.fit(real_train)
            print("✅ PrivBayes fitting completed successfully")
        except Exception as e:
            print(f"❌ Error during PrivBayes fitting: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    elif synthesizer_type.lower() == 'dpctgan':
        print(f"\n🚀 Initializing DP-CTGAN synthesizer...")
        synthesizer = OpacusDifferentiallyPrivateCTGAN(
            epsilon=epsilon,
            delta=1e-5,
            epochs=30,
            batch_size=min(250, len(real_train) // 2),
            verbose=True
        )
        
        # Fit the synthesizer
        print("🔧 Fitting DP-CTGAN on training data...")
        try:
            discrete_cols = preprocessing_metadata.get('discrete_columns', [])
            print(f"📋 Using discrete columns from unified preprocessing: {discrete_cols}")
            synthesizer.fit(real_train, discrete_columns=discrete_cols)
            print("✅ DP-CTGAN fitting completed successfully")
        except Exception as e:
            print(f"❌ Error during DP-CTGAN fitting: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    else:
        raise ValueError(f"Unsupported synthesizer type: {synthesizer_type}")
    
    # Generate synthetic data
    print(f"🎲 Generating {n_synthetic} synthetic samples...")
    try:
        if synthesizer_type.lower() == 'privbayes':
            synthetic_data = synthesizer.sample(n_records=n_synthetic)
        else:  # dpctgan
            synthetic_data = synthesizer.sample(n_samples=n_synthetic)
            
        print(f"✅ Synthetic data generated successfully. Shape: {synthetic_data.shape}")
        return synthetic_data
        
    except Exception as e:
        print(f"❌ Error during synthetic data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_statistical_metrics(real_df, synth_df, results_dir):
    """
    Calculate statistical similarity metrics between real and synthetic data.
    
    Args:
        real_df (pd.DataFrame): Real dataset
        synth_df (pd.DataFrame): Synthetic dataset
        results_dir (str): Directory to save results
    """
    print(f"\n{'=' * 80}")
    print("CALCULATING STATISTICAL METRICS")
    print(f"{'=' * 80}")
    
    metrics = {}
    
    # 1. Distribution Comparison
    print("\n1. 📊 Calculating distribution metrics...")
    for col in real_df.columns:
        if real_df[col].dtype in ['int64', 'float64']:
            # For numerical columns, use Wasserstein distance
            w_dist = wasserstein_distance(real_df[col], synth_df[col])
            metrics[f'wasserstein_{col}'] = w_dist
        else:
            # For categorical columns, use Total Variation Distance
            real_probs = real_df[col].value_counts(normalize=True)
            synth_probs = synth_df[col].value_counts(normalize=True)
            # Align probabilities
            all_categories = set(real_probs.index) | set(synth_probs.index)
            real_probs = real_probs.reindex(all_categories, fill_value=0)
            synth_probs = synth_probs.reindex(all_categories, fill_value=0)
            tvd = 0.5 * np.sum(np.abs(real_probs - synth_probs))
            metrics[f'tvd_{col}'] = tvd
    
    # 2. Correlation Matrix Comparison
    print("\n2. 🔗 Calculating correlation matrices...")
    real_num = real_df.select_dtypes(include=['int64', 'float64'])
    synth_num = synth_df.select_dtypes(include=['int64', 'float64'])
    if real_num.shape[1] > 0 and synth_num.shape[1] > 0:
        real_corr = real_num.corr()
        synth_corr = synth_num.corr()
        corr_diff = np.abs(real_corr - synth_corr)
        metrics['mean_corr_diff'] = corr_diff.mean().mean()
        metrics['max_corr_diff'] = corr_diff.max().max()
        # Save correlation matrices
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_diff, annot=True, cmap='coolwarm', center=0)
        plt.title('Absolute Correlation Difference Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_difference.png'))
        plt.close()
    else:
        print("No numerical columns found. Skipping correlation matrix comparison.")
        metrics['mean_corr_diff'] = None
        metrics['max_corr_diff'] = None
    
    # 3. Propensity Score MSE (PMSE)
    print("\n3. 🎯 Calculating PMSE...")
    # One-hot encode all categorical columns for both real and synthetic data
    real_encoded = pd.get_dummies(real_df, drop_first=True)
    synth_encoded = pd.get_dummies(synth_df, drop_first=True)
    # Align columns
    real_encoded, synth_encoded = real_encoded.align(synth_encoded, join='outer', axis=1, fill_value=0)
    # Combine real and synthetic data
    combined_data = pd.concat([real_encoded, synth_encoded])
    labels = np.array([1] * len(real_df) + [0] * len(synth_df))
    
    # Train a classifier to distinguish between real and synthetic
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(combined_data, labels)
    
    # Get predictions
    preds = clf.predict_proba(combined_data)[:, 1]
    metrics['pmse'] = np.mean((preds - 0.5) ** 2)  # MSE from 0.5 (random guessing)
    metrics['auc'] = roc_auc_score(labels, preds)
    
    # Save metrics
    pd.Series(metrics).to_csv(os.path.join(results_dir, 'statistical_metrics.csv'))
    print(f"✅ Statistical metrics saved to {results_dir}/statistical_metrics.csv")
    
    return metrics

def evaluate_ml_utility(synthetic_data, real_test, real_train, results_dir, target_column='income'):
    """
    Evaluate machine learning utility using TSTR approach.
    
    Args:
        synthetic_data (pd.DataFrame): Synthetic training data
        real_test (pd.DataFrame): Real test data
        real_train (pd.DataFrame): Real training data (for baseline)
        results_dir (str): Directory to save results
        target_column (str): Target column for prediction
    """
    print(f"\n{'=' * 80}")
    print("EVALUATING ML UTILITY (TSTR)")
    print(f"{'=' * 80}")
    
    # Ensure target column is binary numeric (0/1)
    def encode_target(y):
        # Convert to string first to handle any type
        y = y.astype(str)
        # Check for common income patterns
        if set(y.unique()) == {'<=50K', '>50K'}:
            return y.map({'<=50K': 0, '>50K': 1})
        elif set(y.unique()) == {'0', '1'}:
            return y.astype(int)
        elif set(y.unique()) == {'False', 'True'}:
            return y.map({'False': 0, 'True': 1})
        else:
            # Try to infer mapping
            unique_vals = sorted(y.unique())
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                return y.map(mapping)
            else:
                raise ValueError(f"Target column has {len(unique_vals)} unique values, expected 2")
    
    try:
        # Prepare data
        X_synth = synthetic_data.drop(columns=[target_column])
        y_synth = encode_target(synthetic_data[target_column])
        X_real_test = real_test.drop(columns=[target_column])
        y_real_test = encode_target(real_test[target_column])
        X_real_train = real_train.drop(columns=[target_column])
        y_real_train = encode_target(real_train[target_column])
        
        print(f"Target column unique values in synthetic data: {synthetic_data[target_column].unique()}")
        print(f"Encoded target values in synthetic data: {y_synth.unique()}")
        
        # One-hot encode all features and align columns
        X_synth_enc = pd.get_dummies(X_synth, drop_first=True)
        X_real_test_enc = pd.get_dummies(X_real_test, drop_first=True)
        X_real_train_enc = pd.get_dummies(X_real_train, drop_first=True)
        
        # Align all three sets to the same columns
        all_columns = sorted(set(X_synth_enc.columns) | set(X_real_test_enc.columns) | set(X_real_train_enc.columns))
        X_synth_enc = X_synth_enc.reindex(columns=all_columns, fill_value=0)
        X_real_test_enc = X_real_test_enc.reindex(columns=all_columns, fill_value=0)
        X_real_train_enc = X_real_train_enc.reindex(columns=all_columns, fill_value=0)
        
        # Sanitize column names for XGBoost compatibility
        def sanitize_columns(df):
            df.columns = [str(col).replace('[','_').replace(']','_').replace('(','_').replace(')','_').replace('<','_').replace('>','_').replace(' ','_') for col in df.columns]
            return df
        
        X_synth_enc = sanitize_columns(X_synth_enc)
        X_real_test_enc = sanitize_columns(X_real_test_enc)
        X_real_train_enc = sanitize_columns(X_real_train_enc)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42)
        }
        
        results = []
        
        # 1. Baseline (TRTR)
        print("\n1. 📊 Running baseline (TRTR) evaluation...")
        for name, model in models.items():
            print(f"Training {name} on real data...")
            model.fit(X_real_train_enc, y_real_train)
            y_pred = model.predict(X_real_test_enc)
            y_pred_proba = model.predict_proba(X_real_test_enc)[:, 1]
            
            results.append({
                'Model': name,
                'Train Data': 'Real',
                'Accuracy': accuracy_score(y_real_test, y_pred),
                'F1 Score': f1_score(y_real_test, y_pred),
                'AUC': roc_auc_score(y_real_test, y_pred_proba)
            })
        
        # 2. TSTR Evaluation
        print("\n2. 📊 Running TSTR evaluation...")
        for name, model in models.items():
            print(f"Training {name} on synthetic data...")
            model.fit(X_synth_enc, y_synth)
            y_pred = model.predict(X_real_test_enc)
            y_pred_proba = model.predict_proba(X_real_test_enc)[:, 1]
            
            results.append({
                'Model': name,
                'Train Data': 'Synthetic',
                'Accuracy': accuracy_score(y_real_test, y_pred),
                'F1 Score': f1_score(y_real_test, y_pred),
                'AUC': roc_auc_score(y_real_test, y_pred_proba)
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(results_dir, 'ml_utility_results.csv'), index=False)
        print(f"✅ ML utility results saved to {results_dir}/ml_utility_results.csv")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in ML utility evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_privacy_metrics(real_train, real_test, synthetic_data, results_dir):
    """
    Evaluate privacy metrics including membership inference attacks.
    
    Args:
        real_train (pd.DataFrame): Real training data
        real_test (pd.DataFrame): Real test data
        synthetic_data (pd.DataFrame): Synthetic data
        results_dir (str): Directory to save results
    """
    print("\n" + "=" * 80)
    print("EVALUATING PRIVACY METRICS")
    print("=" * 80)

    # 1. Membership inference (distance to closest record)
    print("\n1. 📏 Calculating distances for membership inference...")
    try:
        # Ensure synthetic_data columns match real_train columns
        synthetic_data = synthetic_data[real_train.columns]
        # One-hot encode all columns to ensure numeric distance calculation
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        all_data = pd.concat([real_train, real_test, synthetic_data], axis=0)
        encoder.fit(all_data)
        real_train_encoded = encoder.transform(real_train)
        real_test_encoded = encoder.transform(real_test)
        synth_encoded = encoder.transform(synthetic_data)

        # For each real_train record, find the closest synthetic record
        min_distances_train = []
        for i, row in enumerate(real_train_encoded):
            dists = np.sqrt(np.sum((synth_encoded - row) ** 2, axis=1))
            min_distances_train.append(np.min(dists))
        min_distances_train = np.array(min_distances_train)

        # For each real_test record, find the closest synthetic record
        min_distances_test = []
        for i, row in enumerate(real_test_encoded):
            dists = np.sqrt(np.sum((synth_encoded - row) ** 2, axis=1))
            min_distances_test.append(np.min(dists))
        min_distances_test = np.array(min_distances_test)

        # Concatenate for classifier
        min_distances = np.concatenate([min_distances_train, min_distances_test])
        mean_dist = np.mean(min_distances_train)
        min_dist = np.min(min_distances_train)
        max_dist = np.max(min_distances_train)
        print(f"   📊 Mean distance (train): {mean_dist:.4f}")
        print(f"   📊 Min distance (train): {min_dist:.4f}")
        print(f"   📊 Max distance (train): {max_dist:.4f}")
        # Save to CSV
        pd.DataFrame({
            'mean_distance_train': [mean_dist],
            'min_distance_train': [min_dist],
            'max_distance_train': [max_dist]
        }).to_csv(f"{results_dir}/membership_inference_distances.csv", index=False)
    except Exception as e:
        print(f"❌ Error in membership inference distance calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        min_distances = np.array([])  # Ensure variable is defined for later steps

    # 2. Train membership inference classifier
    print("\n2. 🎯 Training membership inference classifier...")
    is_member = np.array([1] * len(real_train) + [0] * len(real_test))
    clf = LogisticRegression(random_state=42)
    clf.fit(np.array(min_distances).reshape(-1, 1), is_member)

    # Get predictions
    preds = clf.predict_proba(np.array(min_distances).reshape(-1, 1))[:, 1]

    # Calculate metrics
    metrics = {
        'attack_accuracy': accuracy_score(is_member, clf.predict(np.array(min_distances).reshape(-1, 1))),
        'attack_auc': roc_auc_score(is_member, preds),
        'mean_distance_train': mean_dist,
        'min_distance_train': min_dist,
        'max_distance_train': max_dist
    }

    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(f"{results_dir}/privacy_metrics.csv", index=False)
    print("✅ Privacy metrics saved to CSV")

    return metrics

def run_evaluation_pipeline(synthesizer_type='privbayes', train_size=1000, test_size=500, epsilon=1.0, trial=1):
    """
    Run the complete evaluation pipeline for a specified synthesizer.
    
    Args:
        synthesizer_type (str): 'privbayes' or 'dpctgan'
        train_size (int): Number of training samples
        test_size (int): Number of test samples
        epsilon (float): Privacy budget parameter
        trial (int): Trial number for multiple runs
    """
    print(f"🚀 UNIFIED EVALUATION PIPELINE V2")
    print(f"{'=' * 80}")
    print(f"🔧 Synthesizer: {synthesizer_type.upper()}")
    print(f"📊 Training samples: {train_size}")
    print(f"📊 Test samples: {test_size}")
    print(f"🔒 Privacy budget (ε): {epsilon}")
    print(f"🔄 Trial: {trial}")
    print(f"{'=' * 80}")
    
    # Create results directory with epsilon and trial information
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 
                                              f"{synthesizer_type}_eps{epsilon}_trial{trial}"))
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    real_train, real_test, metadata = load_and_preprocess_adult_data(
        synthesizer_type, train_size, test_size
    )
    
    # Step 2: Generate synthetic data
    synthetic_data = test_synthesizer(synthesizer_type, real_train, metadata, epsilon=epsilon)
    if synthetic_data is None:
        print(f"\n❌ Cannot proceed with evaluation due to synthesis failure")
        return
    
    # Step 3: Run all evaluation metrics
    statistical_metrics = calculate_statistical_metrics(real_train, synthetic_data, results_dir)
    ml_utility_results = evaluate_ml_utility(synthetic_data, real_test, real_train, results_dir)
    privacy_metrics = evaluate_privacy_metrics(real_train, real_test, synthetic_data, results_dir)
    
    print(f"\n{'=' * 80}")
    print(f"✅ EVALUATION PIPELINE COMPLETED FOR {synthesizer_type.upper()}")
    print(f"📁 Check '{results_dir}' directory for outputs")
    print(f"{'=' * 80}")

def main():
    """Main function to run evaluation pipeline(s)."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unified evaluation pipeline v2')
    parser.add_argument('--synthesizer', choices=['privbayes', 'dpctgan', 'both'], 
                       default='both', help='Which synthesizer to test')
    parser.add_argument('--train-size', type=int, default=1000, 
                       help='Number of training samples')
    parser.add_argument('--test-size', type=int, default=500, 
                       help='Number of test samples')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials to run for each epsilon value')
    
    args = parser.parse_args()
    
    # Define epsilon values to test
    epsilon_values = [1.0, 0.5, 0.1]
    
    if args.synthesizer == 'both':
        print("🔄 Running evaluation for BOTH synthesizers")
        print("=" * 80)
        
        for epsilon in epsilon_values:
            for trial in range(1, args.trials + 1):
                # Test PrivBayes
                run_evaluation_pipeline('privbayes', args.train_size, args.test_size, epsilon, trial)
                
                print("\n\n")
                
                # Test DP-CTGAN
                run_evaluation_pipeline('dpctgan', args.train_size, args.test_size, epsilon, trial)
                
                print(f"\n{'=' * 80}")
                print(f"🏆 COMPLETED TRIAL {trial} FOR ε={epsilon}")
                print(f"{'=' * 80}")
        
        print(f"\n{'=' * 80}")
        print("🏆 COMPLETE COMPARISON FINISHED")
        print("📊 Check results directory for detailed outputs")
        print(f"{'=' * 80}")
        
    else:
        for epsilon in epsilon_values:
            for trial in range(1, args.trials + 1):
                run_evaluation_pipeline(args.synthesizer, args.train_size, args.test_size, epsilon, trial)
                print(f"\n{'=' * 80}")
                print(f"🏆 COMPLETED TRIAL {trial} FOR ε={epsilon}")
                print(f"{'=' * 80}")

if __name__ == "__main__":
    main() 