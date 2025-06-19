#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from data_preprocessor import AdultDataPreprocessor, CovertypeDataPreprocessor
from synthesizers.privbayes import MyPrivBayes
from synthesizers.dp_ctgan_opacus import OpacusDifferentiallyPrivateCTGAN
from evaluation.statistical_metrics import (
    plot_marginal_distributions,
    plot_correlation_difference,
    calculate_pmse
)
from evaluation.ml_utility import run_tstr_evaluation
from evaluation.privacy_metrics import (
    count_exact_matches,
    calculate_dcr
)

def load_and_preprocess_data(dataset_type, synthesizer_type, train_size=1000, test_size=500):
    """
    Load and preprocess data using the unified preprocessing system.
    
    Args:
        dataset_type (str): Type of dataset ('adult' or 'covertype')
        synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
        train_size (int): Number of training samples to use
        test_size (int): Number of test samples to use
        
    Returns:
        tuple: (real_train, real_test, preprocessing_metadata, original_train, original_test, standard_train, standard_test)
    """
    print(f"=" * 80)
    print(f"LOADING AND PREPROCESSING {dataset_type.upper()} DATA FOR {synthesizer_type.upper()}")
    print(f"=" * 80)
    
    # Initialize preprocessor with absolute path to root data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    if dataset_type.lower() == 'adult':
        preprocessor = AdultDataPreprocessor(data_dir=data_dir)
        # Download data if not exists
        print("📥 Downloading Adult dataset if needed...")
        preprocessor.download()
        
        # Load data
        print("📋 Loading Adult dataset...")
        df_train, df_test = preprocessor.load()
        
        # Combine and split to ensure consistent preprocessing
        print("🔄 Combining and splitting data...")
        df_train_combined, df_test_combined = preprocessor.combine_and_split(
            df_train, df_test, test_size=0.3, random_state=42
        )
        
    elif dataset_type.lower() == 'covertype':
        preprocessor = CovertypeDataPreprocessor(data_dir=data_dir)
        # Download data if not exists
        print("📥 Downloading Covertype dataset if needed...")
        preprocessor.download()
        
        # Load data
        print("📋 Loading Covertype dataset...")
        df = preprocessor.load()
        
        # Split into train and test
        print("🔄 Splitting data...")
        df_train_combined, df_test_combined = preprocessor.combine_and_split(
            df, test_size=0.3, random_state=42
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Store original data for baseline evaluation
    print("📊 Storing original data for baseline evaluation...")
    original_train = df_train_combined.copy()
    original_test = df_test_combined.copy()
    
    # Apply standard preprocessing for baseline evaluation
    print("⚙️  Applying standard preprocessing for baseline evaluation...")
    standard_train = preprocessor.standard_preprocess(df_train_combined)
    standard_test = preprocessor.standard_preprocess(df_test_combined)
    
    # Use unified preprocessing system for synthesizer-specific preprocessing
    print(f"⚙️  Preprocessing data for {synthesizer_type}...")
    df_train_processed, train_metadata = preprocessor.preprocess_for_synthesizer(
        df_train_combined, synthesizer_type
    )
    
    # CRITICAL: Use the same preprocessing for test data to ensure TSTR consistency
    df_test_processed, test_metadata = preprocessor.preprocess_for_synthesizer(
        df_test_combined, synthesizer_type
    )
    
    # Verify metadata consistency
    assert train_metadata == test_metadata, "Train and test preprocessing metadata must match!"
    
    # Take samples for faster testing
    print("📊 Taking samples for faster testing...")
    train_sample = df_train_processed.sample(n=min(train_size, len(df_train_processed)), random_state=42)
    test_sample = df_test_processed.sample(n=min(test_size, len(df_test_processed)), random_state=42)
    
    # Also take corresponding samples from original and standard data
    original_train_sample = original_train.sample(n=min(train_size, len(original_train)), random_state=42)
    original_test_sample = original_test.sample(n=min(test_size, len(original_test)), random_state=42)
    standard_train_sample = standard_train.sample(n=min(train_size, len(standard_train)), random_state=42)
    standard_test_sample = standard_test.sample(n=min(test_size, len(standard_test)), random_state=42)
    
    print(f"✅ Training sample shape: {train_sample.shape}")
    print(f"✅ Test sample shape: {test_sample.shape}")
    print(f"✅ Original training sample shape: {original_train_sample.shape}")
    print(f"✅ Original test sample shape: {original_test_sample.shape}")
    print(f"✅ Standard training sample shape: {standard_train_sample.shape}")
    print(f"✅ Standard test sample shape: {standard_test_sample.shape}")
    print(f"📋 Preprocessing metadata: {train_metadata}")
    
    return (train_sample, test_sample, train_metadata, 
            original_train_sample, original_test_sample, 
            standard_train_sample, standard_test_sample)

def test_synthesizer(synthesizer_type, real_train, preprocessing_metadata, n_synthetic=800, epsilon=1.0):
    """
    Test synthesis for specified synthesizer type.
    
    Args:
        synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
        real_train (pd.DataFrame): Preprocessed training data
        preprocessing_metadata (dict): Metadata from preprocessing
        n_synthetic (int): Number of synthetic samples to generate
        epsilon (float): Privacy budget for differential privacy
        
    Returns:
        pd.DataFrame or None: Generated synthetic data
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING {synthesizer_type.upper()} SYNTHESIS (Epsilon: {epsilon})")
    print(f"{'=' * 80}")
    
    print(f"📊 Real training data shape: {real_train.shape}")
    
    # Show sample of the data
    print(f"\n🔍 Sample of {synthesizer_type} training data:")
    print(real_train.head())
    
    print(f"\n📈 Data types and unique values:")
    for col in real_train.columns:
        unique_count = real_train[col].nunique()
        print(f"   {col}: {unique_count} unique values ({real_train[col].dtype})")
    
    # Initialize synthesizer based on type
    if synthesizer_type.lower() == 'privbayes':
        print(f"\n🚀 Initializing PrivBayes synthesizer with epsilon={epsilon}...")
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
        print(f"\n🚀 Initializing DP-CTGAN synthesizer with epsilon={epsilon}...")
        synthesizer = OpacusDifferentiallyPrivateCTGAN(
            epsilon=epsilon,
            delta=1e-5,
            epochs=30,  # Reduced for faster testing
            batch_size=min(250, len(real_train) // 2),
            verbose=True
        )
        
        # Fit the synthesizer - pass discrete columns from unified preprocessing
        print("🔧 Fitting DP-CTGAN on training data...")
        try:
            discrete_cols = preprocessing_metadata.get('discrete_columns', [])
            print(f"📋 Using discrete columns from unified preprocessing: {discrete_cols}")
            
            # Pass discrete columns to ensure consistency with preprocessing
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
        
        print(f"\n🔍 Sample of synthetic data:")
        print(synthetic_data.head())
        
        # Verify column consistency
        if set(synthetic_data.columns) != set(real_train.columns):
            print(f"⚠️  Column mismatch detected!")
            print(f"   Real columns: {sorted(real_train.columns)}")
            print(f"   Synthetic columns: {sorted(synthetic_data.columns)}")
            # Align columns
            synthetic_data = synthetic_data[real_train.columns]
            print(f"✅ Columns aligned automatically")
        
        return synthetic_data
        
    except Exception as e:
        print(f"❌ Error during synthetic data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_statistical_metrics(real_train, synthetic_data, synthesizer_type, dataset_type, epsilon, trial_num=0, total_trials=1):
    """Test statistical evaluation metrics."""
    print(f"\n{'=' * 80}")
    print(f"TESTING STATISTICAL METRICS - {dataset_type.upper()} - {synthesizer_type.upper()} (Epsilon: {epsilon}) - Trial {trial_num + 1}/{total_trials}")
    print(f"{'=' * 80}")
    
    # Create results directory with dataset and epsilon in path
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 
        f"{dataset_type}_{synthesizer_type}_eps_{epsilon}"))
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = {}
    
    # Test marginal distributions plot
    print(f"\n1. 📊 Testing marginal distribution plots...")
    try:
        # Only save plots for the last trial
        save_path = f"{results_dir}/marginal_distributions.png" if trial_num == total_trials - 1 else None
        plot_marginal_distributions(
            real_df=real_train,
            synth_df=synthetic_data,
            columns=real_train.columns[:6],  # Test with first 6 columns for speed
            save_path=save_path
        )
        if save_path:
            print("✅ Marginal distribution plots generated successfully")
    except Exception as e:
        print(f"❌ Error in marginal distribution plots: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test correlation difference plot
    print(f"\n2. 🔗 Testing correlation difference plot...")
    try:
        # Only save plots for the last trial
        save_path = f"{results_dir}/correlation_difference.png" if trial_num == total_trials - 1 else None
        corr_diff_norm = plot_correlation_difference(
            real_df=real_train,
            synth_df=synthetic_data,
            save_path=save_path
        )
        print(f"✅ Correlation difference plot generated. Frobenius norm: {corr_diff_norm:.4f}")
        metrics['correlation_difference_norm'] = float(corr_diff_norm)
    except Exception as e:
        print(f"❌ Error in correlation difference plot: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test PMSE calculation
    print(f"\n3. 🎯 Testing PMSE calculation...")
    try:
        pmse_score = calculate_pmse(
            real_df=real_train,
            synth_df=synthetic_data
        )
        print(f"✅ PMSE calculated successfully. Score: {pmse_score:.4f}")
        print(f"   Lower PMSE = better synthetic data quality")
        metrics['pmse_score'] = float(pmse_score)
    except Exception as e:
        print(f"❌ Error in PMSE calculation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save metrics to JSON file for this trial
    import json
    metrics_file = os.path.join(results_dir, f'statistical_metrics_trial_{trial_num + 1}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n💾 Statistical metrics for trial {trial_num + 1} saved to {metrics_file}")
    
    return metrics

def test_ml_utility(synthetic_data, real_test, real_train, synthesizer_type, dataset_type, epsilon, 
                   standard_train=None, standard_test=None, target_column='income'):
    """Test ML utility evaluation (TSTR) with proper preprocessing consistency."""
    print(f"\n{'=' * 80}")
    print(f"TESTING ML UTILITY (TSTR) - {dataset_type.upper()} - {synthesizer_type.upper()} (Epsilon: {epsilon})")
    print(f"{'=' * 80}")
    
    print(f"🧠 Testing TSTR evaluation...")
    print(f"   📊 Synthetic training data shape: {synthetic_data.shape}")
    print(f"   📊 Real training data shape: {real_train.shape}")
    print(f"   📊 Real test data shape: {real_test.shape}")
    print(f"   🎯 Target column: {target_column}")
    
    # Use standard preprocessed data for baseline evaluation if provided
    if standard_train is not None and standard_test is not None:
        baseline_train = standard_train
        baseline_test = standard_test
        print("✅ Using standard preprocessed data for baseline evaluation")
    else:
        baseline_train = real_train
        baseline_test = real_test
        print("⚠️  Using synthesizer-specific preprocessed data for baseline evaluation")
    
    # Check if both datasets have the target column
    if target_column not in synthetic_data.columns:
        print(f"❌ Target column '{target_column}' not found in synthetic data")
        print(f"   Available columns: {list(synthetic_data.columns)}")
        return None
    if target_column not in baseline_test.columns:
        print(f"❌ Target column '{target_column}' not found in test data")
        print(f"   Available columns: {list(baseline_test.columns)}")
        return None
    
    # Check if datasets have same columns
    if set(synthetic_data.columns) != set(baseline_test.columns):
        print("❌ Column mismatch between synthetic and test data")
        print(f"   Synthetic columns: {sorted(synthetic_data.columns)}")
        print(f"   Test columns: {sorted(baseline_test.columns)}")
        return None
    
    print("✅ Column consistency verified!")
    
    # Create results directory with dataset and epsilon in path
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 
        f"{dataset_type}_{synthesizer_type}_eps_{epsilon}"))
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = {}
    
    # 1. BASELINE: Train on Real, Test on Real (using standard preprocessing)
    print(f"\n📊 Running BASELINE evaluation: Train on Real, Test on Real...")
    try:
        baseline_results = run_tstr_evaluation(
            synth_train_df=baseline_train,  # Use standard preprocessed real data for training
            real_test_df=baseline_test,     # Use standard preprocessed real data for testing
            target_column=target_column
        )
        print("✅ Baseline evaluation completed successfully")
        print(f"\n📈 BASELINE Results (Train on Real, Test on Real):")
        print(baseline_results.to_string(index=False))
        
        # Save baseline results
        baseline_results.to_csv(f"{results_dir}/baseline_tstr_results.csv", index=False)
        print(f"💾 Baseline results saved to {results_dir}/baseline_tstr_results.csv")
        
        # Store baseline metrics
        metrics['baseline'] = baseline_results.set_index('Model')['Accuracy'].to_dict()
        
    except Exception as e:
        print(f"❌ Error in baseline evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        baseline_results = None
    
    # 2. MAIN EVALUATION: Train on Synthetic, Test on Real (using synthetic data in natural format)
    print(f"\n📊 Running MAIN evaluation: Train on Synthetic, Test on Real...")
    try:
        tstr_results = run_tstr_evaluation(
            synth_train_df=synthetic_data,  # Use synthetic data in its natural format
            real_test_df=baseline_test,     # Use standard preprocessed real data for testing
            target_column=target_column
        )
        print("✅ TSTR evaluation completed successfully")
        print(f"\n📈 TSTR Results (Train on Synthetic, Test on Real):")
        print(tstr_results.to_string(index=False))
        
        # Save main results
        tstr_results.to_csv(f"{results_dir}/tstr_results.csv", index=False)
        print(f"💾 TSTR results saved to {results_dir}/tstr_results.csv")
        
        # Store TSTR metrics
        metrics['tstr'] = tstr_results.set_index('Model')['Accuracy'].to_dict()
        
        # 3. COMPARISON: Show utility retention
        if baseline_results is not None:
            print(f"\n📊 UTILITY RETENTION COMPARISON:")
            print("=" * 50)
            retention_metrics = {}
            for _, baseline_row in baseline_results.iterrows():
                tstr_row = tstr_results[tstr_results['Model'] == baseline_row['Model']]
                if not tstr_row.empty:
                    baseline_acc = baseline_row['Accuracy']
                    tstr_acc = tstr_row.iloc[0]['Accuracy']
                    retention = (tstr_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
                    print(f"   {baseline_row['Model']:15} | Baseline: {baseline_acc:.3f} | Synthetic: {tstr_acc:.3f} | Retention: {retention:.1f}%")
                    retention_metrics[baseline_row['Model']] = retention
            print("=" * 50)
            metrics['retention'] = retention_metrics
        
    except Exception as e:
        print(f"❌ Error in TSTR evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return metrics

def test_privacy_metrics(real_train, synthetic_data, synthesizer_type, dataset_type, epsilon, trial_num=0, total_trials=1):
    """Test privacy evaluation metrics."""
    print(f"\n{'=' * 80}")
    print(f"TESTING PRIVACY METRICS - {dataset_type.upper()} - {synthesizer_type.upper()} (Epsilon: {epsilon}) - Trial {trial_num + 1}/{total_trials}")
    print(f"{'=' * 80}")
    
    # Create results directory with dataset and epsilon in path
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 
        f"{dataset_type}_{synthesizer_type}_eps_{epsilon}"))
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = {}
    
    # Test exact matches
    print(f"\n1. 🔒 Testing exact matches count...")
    try:
        exact_matches = count_exact_matches(
            real_df=real_train,
            synth_df=synthetic_data
        )
        match_percentage = (exact_matches / len(synthetic_data)) * 100
        print(f"✅ Exact matches counted successfully")
        print(f"   🔢 Number of exact matches: {exact_matches}")
        print(f"   📊 Percentage of synthetic data that are exact matches: {match_percentage:.2f}%")
        metrics['exact_matches'] = int(exact_matches)
        metrics['exact_matches_percentage'] = float(match_percentage)
    except Exception as e:
        print(f"❌ Error in counting exact matches: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test DCR calculation
    print(f"\n2. 📏 Testing DCR (Distance to Closest Record)...")
    try:
        # Ensure both datasets have the same column order
        synth_aligned = synthetic_data[real_train.columns]
        
        dcr_stats = calculate_dcr(
            synth_df=synth_aligned,
            real_df=real_train
        )
        print("✅ DCR calculated successfully")
        print(f"   📊 Mean distance: {dcr_stats['mean']:.4f}")
        print(f"   📊 Min distance: {dcr_stats['min']:.4f}")
        print(f"   📊 Max distance: {dcr_stats['max']:.4f}")
        metrics['dcr_stats'] = {
            'mean': float(dcr_stats['mean']),
            'min': float(dcr_stats['min']),
            'max': float(dcr_stats['max'])
        }
    except Exception as e:
        print(f"❌ Error in DCR calculation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save metrics to JSON file for this trial
    import json
    metrics_file = os.path.join(results_dir, f'privacy_metrics_trial_{trial_num + 1}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n💾 Privacy metrics for trial {trial_num + 1} saved to {metrics_file}")
    
    return metrics

def aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple trials."""
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics:
                if isinstance(metrics[key], dict):
                    # Handle nested metrics (like dcr_stats or ml metrics)
                    if key not in aggregated:
                        aggregated[key] = {}
                    for subkey, subvalue in metrics[key].items():
                        if subkey not in aggregated[key]:
                            aggregated[key][subkey] = []
                        aggregated[key][subkey].append(subvalue)
                else:
                    values.append(metrics[key])
        
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        elif key in aggregated:
            # Handle nested metrics
            for subkey in aggregated[key]:
                values = aggregated[key][subkey]
                aggregated[key][subkey] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
    
    return aggregated

def run_evaluation_pipeline(dataset_type, synthesizer_type, train_size=1000, test_size=500, epsilon_values=[1.0], n_trials=3):
    """
    Run the complete evaluation pipeline for a specified synthesizer.
    
    Args:
        dataset_type (str): Type of dataset ('adult' or 'covertype')
        synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
        train_size (int): Number of training samples
        test_size (int): Number of test samples
        epsilon_values (list): List of privacy budgets for differential privacy
        n_trials (int): Number of trials to run
    """
    print(f"🚀 UNIFIED EVALUATION PIPELINE")
    print(f"{'=' * 80}")
    print(f"📊 Dataset: {dataset_type.upper()}")
    print(f"🔧 Synthesizer: {synthesizer_type.upper()}")
    print(f"🔒 Epsilon values: {epsilon_values}")
    print(f"📊 Training samples: {train_size}")
    print(f"📊 Test samples: {test_size}")
    print(f"🔄 Number of trials: {n_trials}")
    print(f"{'=' * 80}")
    
    # Run evaluation for each epsilon value
    for epsilon in epsilon_values:
        print(f"\n{'=' * 80}")
        print(f"RUNNING EVALUATION FOR EPSILON = {epsilon}")
        print(f"{'=' * 80}")
        
        # Create results directory
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 
            f"{dataset_type}_{synthesizer_type}_eps_{epsilon}"))
        os.makedirs(results_dir, exist_ok=True)
        
        # Store metrics for all trials
        all_statistical_metrics = []
        all_privacy_metrics = []
        all_ml_metrics = []
        
        for trial in range(n_trials):
            print(f"\n{'=' * 80}")
            print(f"RUNNING TRIAL {trial + 1}/{n_trials}")
            print(f"{'=' * 80}")
            
            # Step 1: Load and preprocess data using unified system
            real_train, real_test, metadata, original_train, original_test, standard_train, standard_test = load_and_preprocess_data(dataset_type, synthesizer_type, train_size, test_size)
            
            # Step 2: Test synthesizer
            synthetic_data = test_synthesizer(synthesizer_type, real_train, metadata, epsilon=epsilon)
            if synthetic_data is None:
                print(f"\n❌ Cannot proceed with evaluation due to synthesis failure")
                continue
            
            # Step 3: Run all evaluation metrics
            statistical_metrics = test_statistical_metrics(real_train, synthetic_data, synthesizer_type, dataset_type, epsilon, trial, n_trials)
            privacy_metrics = test_privacy_metrics(real_train, synthetic_data, synthesizer_type, dataset_type, epsilon, trial, n_trials)
            # Use appropriate target column based on dataset type
            target_column = 'Cover_Type' if dataset_type.lower() == 'covertype' else 'income'
            ml_metrics = test_ml_utility(synthetic_data, real_test, real_train, synthesizer_type, dataset_type, epsilon, standard_train, standard_test, target_column=target_column)
            
            all_statistical_metrics.append(statistical_metrics)
            all_privacy_metrics.append(privacy_metrics)
            if ml_metrics is not None:
                all_ml_metrics.append(ml_metrics)
        
        # Aggregate metrics
        aggregated_statistical = aggregate_metrics(all_statistical_metrics)
        aggregated_privacy = aggregate_metrics(all_privacy_metrics)
        aggregated_ml = aggregate_metrics(all_ml_metrics) if all_ml_metrics else None
        
        # Save aggregated metrics
        import json
        with open(os.path.join(results_dir, 'aggregated_statistical_metrics.json'), 'w') as f:
            json.dump(aggregated_statistical, f, indent=4)
        with open(os.path.join(results_dir, 'aggregated_privacy_metrics.json'), 'w') as f:
            json.dump(aggregated_privacy, f, indent=4)
        if aggregated_ml:
            with open(os.path.join(results_dir, 'aggregated_ml_metrics.json'), 'w') as f:
                json.dump(aggregated_ml, f, indent=4)
        
        print(f"\n{'=' * 80}")
        print(f"✅ EVALUATION PIPELINE COMPLETED FOR {dataset_type.upper()} - {synthesizer_type.upper()} (Epsilon: {epsilon})")
        print(f"📊 Results aggregated across {n_trials} trials")
        print(f"📁 Check 'results/{dataset_type}_{synthesizer_type}_eps_{epsilon}/' directory for outputs")
        print(f"{'=' * 80}")

def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Run synthetic data evaluation pipeline')
    parser.add_argument('--synthesizer', type=str, default='privbayes',
                      choices=['privbayes', 'dpctgan'],
                      help='Type of synthesizer to use (default: privbayes)')
    parser.add_argument('--dataset', type=str, default='adult',
                      choices=['adult', 'covertype'],
                      help='Dataset to use (default: adult)')
    parser.add_argument('--train-size', type=int, default=1000,
                      help='Number of training samples to use (default: 1000)')
    parser.add_argument('--test-size', type=int, default=500,
                      help='Number of test samples to use (default: 500)')
    parser.add_argument('--epsilon', type=str, default='1.0',
                      help='Privacy budget(s) for differential privacy. Can be a single value or comma-separated list (default: 1.0)')
    parser.add_argument('--n-trials', type=int, default=3,
                      help='Number of trials to run (default: 3)')
    
    args = parser.parse_args()
    
    # Parse epsilon values
    try:
        epsilon_values = [float(eps.strip()) for eps in args.epsilon.split(',')]
    except ValueError:
        print("Error: Epsilon values must be comma-separated numbers")
        return
    
    # Run the evaluation pipeline with all epsilon values
    run_evaluation_pipeline(
        dataset_type=args.dataset,
        synthesizer_type=args.synthesizer,
        train_size=args.train_size,
        test_size=args.test_size,
        epsilon_values=epsilon_values,
        n_trials=args.n_trials
    )

if __name__ == "__main__":
    main() 