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

def load_and_preprocess_adult_data():
    """Load and preprocess the Adult dataset using the AdultDataPreprocessor."""
    
    # Initialize preprocessor
    preprocessor = AdultDataPreprocessor()
    
    # Download data if not exists
    print("Downloading Adult dataset if needed...")
    preprocessor.download()
    
    # Load data
    print("Loading Adult dataset...")
    df_train, df_test = preprocessor.load()
    
    # Combine and split to ensure consistent preprocessing
    print("Combining and splitting data...")
    df_train_combined, df_test_combined = preprocessor.combine_and_split(
        df_train, df_test, test_size=0.3, random_state=42
    )
    
    # Preprocess for PrivBayes (converts to categorical)
    print("Preprocessing data for PrivBayes...")
    df_train_privbayes = preprocessor.preprocess_for_privbayes(df_train_combined)
    
    # For test data, use the same PrivBayes preprocessing to match synthetic data format
    df_test_privbayes = preprocessor.preprocess_for_privbayes(df_test_combined)
    
    # Take smaller samples for faster testing
    print("Taking samples for faster testing...")
    train_sample_privbayes = df_train_privbayes.sample(n=1000, random_state=42)
    test_sample_privbayes = df_test_privbayes.sample(n=500, random_state=42)
    
    print(f"PrivBayes training sample shape: {train_sample_privbayes.shape}")
    print(f"PrivBayes test sample shape: {test_sample_privbayes.shape}")
    
    return train_sample_privbayes, test_sample_privbayes

def test_privbayes_synthesis():
    """Test PrivBayes synthetic data generation."""
    print("=" * 60)
    print("TESTING PRIVBAYES SYNTHESIS")
    print("=" * 60)
    
    # Load data
    real_train_privbayes, real_test_privbayes = load_and_preprocess_adult_data()
    
    print(f"Real training data shape: {real_train_privbayes.shape}")
    print(f"Real test data shape: {real_test_privbayes.shape}")
    
    # Show sample of the data
    print("\nSample of PrivBayes training data:")
    print(real_train_privbayes.head())
    
    print("\nData types and unique values:")
    for col in real_train_privbayes.columns:
        unique_count = real_train_privbayes[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    # Initialize PrivBayes
    print("\nInitializing PrivBayes synthesizer...")
    synthesizer = MyPrivBayes(epsilon=1.0)
    
    # Fit the synthesizer
    print("Fitting PrivBayes on training data...")
    try:
        synthesizer.fit(real_train_privbayes)
        print("✓ PrivBayes fitting completed successfully")
    except Exception as e:
        print(f"✗ Error during PrivBayes fitting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Generate synthetic data
    print("Generating synthetic data...")
    try:
        synthetic_data = synthesizer.sample(n_records=800)
        print(f"✓ Synthetic data generated successfully. Shape: {synthetic_data.shape}")
        
        print("\nSample of synthetic data:")
        print(synthetic_data.head())
        
    except Exception as e:
        print(f"✗ Error during synthetic data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    return real_train_privbayes, real_test_privbayes, synthetic_data

def test_statistical_metrics(real_train, synthetic_data):
    """Test statistical evaluation metrics."""
    print("\n" + "=" * 60)
    print("TESTING STATISTICAL METRICS")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Test marginal distributions plot
    print("\n1. Testing marginal distribution plots...")
    try:
        plot_marginal_distributions(
            real_df=real_train,
            synth_df=synthetic_data,
            columns=real_train.columns[:6],  # Test with first 6 columns for speed
            save_path="results/marginal_distributions_test.png"
        )
        print("✓ Marginal distribution plots generated successfully")
    except Exception as e:
        print(f"✗ Error in marginal distribution plots: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test correlation difference plot
    print("\n2. Testing correlation difference plot...")
    try:
        corr_diff_norm = plot_correlation_difference(
            real_df=real_train,
            synth_df=synthetic_data,
            save_path="results/correlation_difference_test.png"
        )
        print(f"✓ Correlation difference plot generated successfully. Frobenius norm: {corr_diff_norm:.4f}")
    except Exception as e:
        print(f"✗ Error in correlation difference plot: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test PMSE calculation
    print("\n3. Testing PMSE calculation...")
    try:
        pmse_score = calculate_pmse(
            real_df=real_train,
            synth_df=synthetic_data
        )
        print(f"✓ PMSE calculated successfully. Score: {pmse_score:.4f}")
    except Exception as e:
        print(f"✗ Error in PMSE calculation: {str(e)}")
        import traceback
        traceback.print_exc()

def test_ml_utility(synthetic_data, real_test, target_column='income'):
    """Test ML utility evaluation (TSTR)."""
    print("\n" + "=" * 60)
    print("TESTING ML UTILITY (TSTR)")
    print("=" * 60)
    
    print("\nTesting TSTR evaluation with PrivBayes categorical data...")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Real test data shape: {real_test.shape}")
    
    # Check if both datasets have the target column
    if target_column not in synthetic_data.columns:
        print(f"✗ Target column '{target_column}' not found in synthetic data")
        print(f"Available columns in synthetic data: {list(synthetic_data.columns)}")
        return
    if target_column not in real_test.columns:
        print(f"✗ Target column '{target_column}' not found in real test data")
        print(f"Available columns in test data: {list(real_test.columns)}")
        return
    
    # Check if datasets have same columns
    if set(synthetic_data.columns) != set(real_test.columns):
        print("✗ Column mismatch between synthetic and test data")
        print(f"Synthetic columns: {sorted(synthetic_data.columns)}")
        print(f"Test columns: {sorted(real_test.columns)}")
        return
    
    try:
        tstr_results = run_tstr_evaluation(
            synth_train_df=synthetic_data,
            real_test_df=real_test,
            target_column=target_column
        )
        print("✓ TSTR evaluation completed successfully")
        print("\nTSTR Results:")
        print(tstr_results.to_string(index=False))
    except Exception as e:
        print(f"✗ Error in TSTR evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def test_privacy_metrics(real_train, synthetic_data):
    """Test privacy evaluation metrics."""
    print("\n" + "=" * 60)
    print("TESTING PRIVACY METRICS")
    print("=" * 60)
    
    # Test exact matches
    print("\n1. Testing exact matches count...")
    try:
        exact_matches = count_exact_matches(
            real_df=real_train,
            synth_df=synthetic_data
        )
        match_percentage = (exact_matches / len(synthetic_data)) * 100
        print(f"✓ Exact matches counted successfully")
        print(f"   Number of exact matches: {exact_matches}")
        print(f"   Percentage of synthetic data that are exact matches: {match_percentage:.2f}%")
    except Exception as e:
        print(f"✗ Error in counting exact matches: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test DCR calculation
    print("\n2. Testing DCR (Distance to Closest Record)...")
    try:
        # Ensure both datasets have the same column order
        synth_aligned = synthetic_data[real_train.columns]
        
        dcr_stats = calculate_dcr(
            synth_df=synth_aligned,
            real_df=real_train
        )
        print("✓ DCR calculated successfully")
        print(f"   Mean distance: {dcr_stats['mean']:.4f}")
        print(f"   Min distance: {dcr_stats['min']:.4f}")
        print(f"   Max distance: {dcr_stats['max']:.4f}")
    except Exception as e:
        print(f"✗ Error in DCR calculation: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all tests."""
    print("COMPREHENSIVE EVALUATION PIPELINE TEST")
    print("=" * 60)
    print("This script will test the PrivBayes synthesizer and all evaluation metrics")
    print("=" * 60)
    
    # Test PrivBayes synthesis
    result = test_privbayes_synthesis()
    if result[0] is None or result[2] is None:
        print("\n✗ Cannot proceed with evaluation tests due to synthesis failure")
        return
    
    real_train_privbayes, real_test_privbayes, synthetic_data = result
    
    # Test all evaluation metrics
    test_statistical_metrics(real_train_privbayes, synthetic_data)
    test_ml_utility(synthetic_data, real_test_privbayes)
    test_privacy_metrics(real_train_privbayes, synthetic_data)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("Check the 'results' directory for generated plots and outputs.")

if __name__ == "__main__":
    main() 