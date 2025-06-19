import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

def normalize_mixed_types(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple:
    """
    Normalize data types between real and synthetic dataframes to handle mixed types.
    This ensures consistent data types for sklearn preprocessing.
    
    Args:
        real_df (pd.DataFrame): Real data
        synth_df (pd.DataFrame): Synthetic data
        
    Returns:
        tuple: (normalized_real_df, normalized_synth_df)
    """
    real_normalized = real_df.copy()
    synth_normalized = synth_df.copy()
    
    for col in real_df.columns:
        real_col_type = real_df[col].dtype
        
        if real_col_type in ['object', 'category'] or real_df[col].nunique() <= 20:
            # Treat as categorical - convert everything to strings
            real_normalized[col] = real_normalized[col].astype(str)
            synth_normalized[col] = synth_normalized[col].astype(str)
        else:
            # Treat as numerical - convert to numeric, handle errors
            real_normalized[col] = pd.to_numeric(real_normalized[col], errors='coerce')
            synth_normalized[col] = pd.to_numeric(synth_normalized[col], errors='coerce')
            
            # Fill NaN values with median
            median_val = real_normalized[col].median()
            real_normalized[col] = real_normalized[col].fillna(median_val)
            synth_normalized[col] = synth_normalized[col].fillna(median_val)
    
    return real_normalized, synth_normalized

def run_tstr_evaluation(
    synth_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Run Train on Synthetic, Test on Real (TSTR) evaluation.
    
    Args:
        synth_train_df (pd.DataFrame): Synthetic training data
        real_test_df (pd.DataFrame): Real test data
        target_column (str): Name of the target column
        
    Returns:
        pd.DataFrame: Results containing model performance metrics
    """
    # Normalize data types to handle mixed types from DP-CTGAN
    synth_train_norm, real_test_norm = normalize_mixed_types(synth_train_df, real_test_df)
    
    # Separate features and target
    X_train = synth_train_norm.drop(target_column, axis=1)
    y_train = synth_train_norm[target_column]
    X_test = real_test_norm.drop(target_column, axis=1)
    y_test = real_test_norm[target_column]
    
    # Re-identify categorical and numerical columns after normalization
    categorical_cols = []
    numerical_cols = []
    
    for col in X_train.columns:
        if X_train[col].dtype in ['object', 'category'] or X_train[col].nunique() <= 20:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Create preprocessing pipeline
    transformers = []
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    if numerical_cols:
        transformers.append(('num', StandardScaler(), numerical_cols))
    
    if not transformers:
        raise ValueError("No valid columns for preprocessing")
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Define classifiers to test
    classifiers: Dict[str, Any] = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100,))
    }
    
    # Store results
    results = []
    
    # Train and evaluate each classifier
    for name, classifier in classifiers.items():
        # Create pipeline with preprocessing and classifier
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # Train on synthetic data
        pipeline.fit(X_train, y_train)
        
        # Make predictions on real test data
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1-Score': f1
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Round metrics to 4 decimal places
    results_df['Accuracy'] = results_df['Accuracy'].round(4)
    results_df['F1-Score'] = results_df['F1-Score'].round(4)
    
    return results_df
