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
    # Create copies to avoid modifying original dataframes
    synth_train_df = synth_train_df.copy()
    real_test_df = real_test_df.copy()
    
    # Separate features and target
    X_train = synth_train_df.drop(target_column, axis=1)
    y_train = synth_train_df[target_column]
    X_test = real_test_df.drop(target_column, axis=1)
    y_test = real_test_df[target_column]
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    
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
