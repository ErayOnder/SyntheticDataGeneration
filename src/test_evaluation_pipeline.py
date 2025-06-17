#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from data_preprocessor import AdultDataPreprocessor
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
    
    # Combine and split to ensure consistent preprocessing
    print("🔄 Combining and splitting data...")
    df_train_combined, df_test_combined = preprocessor.combine_and_split(
        df_train, df_test, test_size=0.3, random_state=42
    )
    
    # Use unified preprocessing system
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
    
    print(f"✅ Training sample shape: {train_sample.shape}")
    print(f"✅ Test sample shape: {test_sample.shape}")
    print(f"📋 Preprocessing metadata: {train_metadata}")
    
    return train_sample, test_sample, train_metadata

def test_synthesizer(synthesizer_type, real_train, preprocessing_metadata, n_synthetic=800):
    """
    Test synthesis for specified synthesizer type.
    
    Args:
        synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
        real_train (pd.DataFrame): Preprocessed training data
        preprocessing_metadata (dict): Metadata from preprocessing
        n_synthetic (int): Number of synthetic samples to generate
        
    Returns:
        pd.DataFrame or None: Generated synthetic data
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING {synthesizer_type.upper()} SYNTHESIS")
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
        print(f"\n🚀 Initializing PrivBayes synthesizer...")
        synthesizer = MyPrivBayes(epsilon=1.0)
        
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
            epsilon=1.0,
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

def test_statistical_metrics(real_train, synthetic_data, synthesizer_type):
    """Test statistical evaluation metrics."""
    print(f"\n{'=' * 80}")
    print(f"TESTING STATISTICAL METRICS - {synthesizer_type.upper()}")
    print(f"{'=' * 80}")
    
    # Create results directory in root, not src
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', synthesizer_type))
    os.makedirs(results_dir, exist_ok=True)
    
    # Test marginal distributions plot
    print(f"\n1. 📊 Testing marginal distribution plots...")
    try:
        plot_marginal_distributions(
            real_df=real_train,
            synth_df=synthetic_data,
            columns=real_train.columns[:6],  # Test with first 6 columns for speed
            save_path=f"{results_dir}/marginal_distributions.png"
        )
        print("✅ Marginal distribution plots generated successfully")
    except Exception as e:
        print(f"❌ Error in marginal distribution plots: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test correlation difference plot
    print(f"\n2. 🔗 Testing correlation difference plot...")
    try:
        corr_diff_norm = plot_correlation_difference(
            real_df=real_train,
            synth_df=synthetic_data,
            save_path=f"{results_dir}/correlation_difference.png"
        )
        print(f"✅ Correlation difference plot generated. Frobenius norm: {corr_diff_norm:.4f}")
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
    except Exception as e:
        print(f"❌ Error in PMSE calculation: {str(e)}")
        import traceback
        traceback.print_exc()

def test_ml_utility(synthetic_data, real_test, real_train, synthesizer_type, target_column='income'):
    """Test ML utility evaluation (TSTR) with proper preprocessing consistency."""
    print(f"\n{'=' * 80}")
    print(f"TESTING ML UTILITY (TSTR) - {synthesizer_type.upper()}")
    print(f"{'=' * 80}")
    
    print(f"🧠 Testing TSTR evaluation...")
    print(f"   📊 Synthetic training data shape: {synthetic_data.shape}")
    print(f"   📊 Real training data shape: {real_train.shape}")
    print(f"   📊 Real test data shape: {real_test.shape}")
    print(f"   🎯 Target column: {target_column}")
    
    # Check if both datasets have the target column
    if target_column not in synthetic_data.columns:
        print(f"❌ Target column '{target_column}' not found in synthetic data")
        print(f"   Available columns: {list(synthetic_data.columns)}")
        return
    if target_column not in real_test.columns:
        print(f"❌ Target column '{target_column}' not found in real test data")
        print(f"   Available columns: {list(real_test.columns)}")
        return
    
    # Check if datasets have same columns (this should be guaranteed by unified preprocessing)
    if set(synthetic_data.columns) != set(real_test.columns):
        print("❌ Column mismatch between synthetic and test data")
        print(f"   Synthetic columns: {sorted(synthetic_data.columns)}")
        print(f"   Test columns: {sorted(real_test.columns)}")
        return
    
    print("✅ Column consistency verified - unified preprocessing working correctly!")
    
    # Create results directory in root, not src
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', synthesizer_type))
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. BASELINE: Train on Real, Test on Real
    print(f"\n📊 Running BASELINE evaluation: Train on Real, Test on Real...")
    try:
        baseline_results = run_tstr_evaluation(
            synth_train_df=real_train,  # Use real data for training
            real_test_df=real_test,
            target_column=target_column
        )
        print("✅ Baseline evaluation completed successfully")
        print(f"\n📈 BASELINE Results (Train on Real, Test on Real):")
        print(baseline_results.to_string(index=False))
        
        # Save baseline results
        baseline_results.to_csv(f"{results_dir}/baseline_tstr_results.csv", index=False)
        print(f"💾 Baseline results saved to {results_dir}/baseline_tstr_results.csv")
        
    except Exception as e:
        print(f"❌ Error in baseline evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        baseline_results = None
    
    # 2. MAIN EVALUATION: Train on Synthetic, Test on Real
    print(f"\n📊 Running MAIN evaluation: Train on Synthetic, Test on Real...")
    try:
        tstr_results = run_tstr_evaluation(
            synth_train_df=synthetic_data,
            real_test_df=real_test,
            target_column=target_column
        )
        print("✅ TSTR evaluation completed successfully")
        print(f"\n📈 TSTR Results (Train on Synthetic, Test on Real):")
        print(tstr_results.to_string(index=False))
        
        # Save main results
        tstr_results.to_csv(f"{results_dir}/tstr_results.csv", index=False)
        print(f"💾 TSTR results saved to {results_dir}/tstr_results.csv")
        
        # 3. COMPARISON: Show utility retention
        if baseline_results is not None:
            print(f"\n📊 UTILITY RETENTION COMPARISON:")
            print("=" * 50)
            for _, baseline_row in baseline_results.iterrows():
                tstr_row = tstr_results[tstr_results['Model'] == baseline_row['Model']]
                if not tstr_row.empty:
                    baseline_acc = baseline_row['Accuracy']
                    tstr_acc = tstr_row.iloc[0]['Accuracy']
                    retention = (tstr_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
                    print(f"   {baseline_row['Model']:15} | Baseline: {baseline_acc:.3f} | Synthetic: {tstr_acc:.3f} | Retention: {retention:.1f}%")
            print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error in TSTR evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def test_privacy_metrics(real_train, synthetic_data, synthesizer_type):
    """Test privacy evaluation metrics."""
    print(f"\n{'=' * 80}")
    print(f"TESTING PRIVACY METRICS - {synthesizer_type.upper()}")
    print(f"{'=' * 80}")
    
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
    except Exception as e:
        print(f"❌ Error in DCR calculation: {str(e)}")
        import traceback
        traceback.print_exc()

def run_evaluation_pipeline(synthesizer_type='privbayes', train_size=1000, test_size=500):
    """
    Run the complete evaluation pipeline for a specified synthesizer.
    
    Args:
        synthesizer_type (str): 'privbayes' or 'dpctgan'
        train_size (int): Number of training samples
        test_size (int): Number of test samples
    """
    print(f"🚀 UNIFIED EVALUATION PIPELINE")
    print(f"{'=' * 80}")
    print(f"🔧 Synthesizer: {synthesizer_type.upper()}")
    print(f"📊 Training samples: {train_size}")
    print(f"📊 Test samples: {test_size}")
    print(f"{'=' * 80}")
    
    # Step 1: Load and preprocess data using unified system
    real_train, real_test, metadata = load_and_preprocess_adult_data(
        synthesizer_type, train_size, test_size
    )
    
    # Step 2: Test synthesizer
    synthetic_data = test_synthesizer(synthesizer_type, real_train, metadata)
    if synthetic_data is None:
        print(f"\n❌ Cannot proceed with evaluation due to synthesis failure")
        return
    
    # Step 3: Run all evaluation metrics
    test_statistical_metrics(real_train, synthetic_data, synthesizer_type)
    test_ml_utility(synthetic_data, real_test, real_train, synthesizer_type)
    test_privacy_metrics(real_train, synthetic_data, synthesizer_type)
    
    print(f"\n{'=' * 80}")
    print(f"✅ EVALUATION PIPELINE COMPLETED FOR {synthesizer_type.upper()}")
    print(f"{'=' * 80}")
    print(f"📁 Check 'results/{synthesizer_type}/' directory for outputs")

def main():
    """Main function to run evaluation pipeline(s)."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unified evaluation pipeline')
    parser.add_argument('--synthesizer', choices=['privbayes', 'dpctgan', 'both'], 
                       default='both', help='Which synthesizer to test')
    parser.add_argument('--train-size', type=int, default=1000, 
                       help='Number of training samples')
    parser.add_argument('--test-size', type=int, default=500, 
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    if args.synthesizer == 'both':
        print("🔄 Running evaluation for BOTH synthesizers")
        print("=" * 80)
        
        # Test PrivBayes
        run_evaluation_pipeline('privbayes', args.train_size, args.test_size)
        
        print("\n\n")
        
        # Test DP-CTGAN
        run_evaluation_pipeline('dpctgan', args.train_size, args.test_size)
        
        print(f"\n{'=' * 80}")
        print("🏆 COMPLETE COMPARISON FINISHED")
        print("📊 Check results/privbayes/ and results/dpctgan/ for detailed outputs")
        print(f"{'=' * 80}")
        
    else:
        run_evaluation_pipeline(args.synthesizer, args.train_size, args.test_size)

if __name__ == "__main__":
    main() 