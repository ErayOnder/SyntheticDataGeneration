import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        
        # Check if column should be categorical or numerical
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

def plot_marginal_distributions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Create a grid of plots comparing marginal distributions between real and synthetic data.
    
    Args:
        real_df (pd.DataFrame): DataFrame containing real data
        synth_df (pd.DataFrame): DataFrame containing synthetic data
        columns (List[str]): List of column names to plot
        save_path (Optional[str]): Path to save the plot. If None, plot is not saved or displayed
    """
    # Calculate grid dimensions
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each column
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        if real_df[col].dtype in ['object', 'category']:
            # Categorical data - bar plots
            real_counts = real_df[col].value_counts(normalize=True)
            synth_counts = synth_df[col].value_counts(normalize=True)
            
            # Get all unique categories
            all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
            
            # Create bar positions
            x = np.arange(len(all_categories))
            width = 0.35
            
            # Plot bars
            ax.bar(x - width/2, [real_counts.get(cat, 0) for cat in all_categories],
                  width, label='Real', alpha=0.7)
            ax.bar(x + width/2, [synth_counts.get(cat, 0) for cat in all_categories],
                  width, label='Synthetic', alpha=0.7)
            
            # Customize x-axis
            ax.set_xticks(x)
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            
        else:
            # Numerical data - KDE plots
            sns.kdeplot(data=real_df[col], ax=ax, label='Real', alpha=0.7)
            sns.kdeplot(data=synth_df[col], ax=ax, label='Synthetic', alpha=0.7)
        
        # Customize plot
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density' if real_df[col].dtype not in ['object', 'category'] else 'Proportion')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Close the plot to free memory
    plt.close()

def plot_correlation_difference(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> float:
    """
    Plot the difference in correlation matrices between real and synthetic data.
    
    Args:
        real_df (pd.DataFrame): DataFrame containing real data
        synth_df (pd.DataFrame): DataFrame containing synthetic data
        save_path (Optional[str]): Path to save the plot. If None, plot is not saved or displayed
        
    Returns:
        float: Frobenius norm of the correlation difference matrix
    """
    # Create copies to avoid modifying original dataframes
    real_df = real_df.copy()
    synth_df = synth_df.copy()
    
    # Convert categorical columns to numerical codes
    for col in real_df.columns:
        if real_df[col].dtype in ['object', 'category']:
            # Get all unique categories from both dataframes
            all_categories = sorted(set(real_df[col].unique()) | set(synth_df[col].unique()))
            # Create a mapping dictionary
            cat_map = {cat: idx for idx, cat in enumerate(all_categories)}
            # Convert to numerical codes
            real_df[col] = real_df[col].map(cat_map)
            synth_df[col] = synth_df[col].map(cat_map)
    
    # Compute correlation matrices
    real_corr = real_df.corr(method='pearson')
    synth_corr = synth_df.corr(method='pearson')
    
    # Calculate absolute difference matrix
    diff_matrix = np.abs(real_corr - synth_corr)
    
    # Calculate Frobenius norm
    frobenius_norm = float(np.sqrt(np.sum(diff_matrix.values ** 2)))
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Absolute Correlation Difference'}
    )
    
    plt.title('Absolute Difference in Correlation Matrices\n(Real vs Synthetic)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Close the plot to free memory
    plt.close()
    
    return frobenius_norm

def calculate_pmse(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    random_state: int = 42
) -> float:
    """
    Calculate the Propensity Mean Squared Error (PMSE) between real and synthetic data.
    
    PMSE measures how distinguishable the synthetic data is from the real data.
    A lower PMSE indicates better synthetic data quality.
    
    Args:
        real_df (pd.DataFrame): DataFrame containing real data
        synth_df (pd.DataFrame): DataFrame containing synthetic data
        random_state (int): Random seed for reproducibility
        
    Returns:
        float: PMSE score (lower is better)
    """
    # Normalize data types to handle mixed types from DP-CTGAN
    real_df_norm, synth_df_norm = normalize_mixed_types(real_df, synth_df)
    
    # Add source indicator
    real_df_norm['is_real'] = 1
    synth_df_norm['is_real'] = 0
    
    # Combine datasets
    combined_df = pd.concat([real_df_norm, synth_df_norm], axis=0, ignore_index=True)
    
    # Separate features and target
    X = combined_df.drop('is_real', axis=1)
    y = combined_df['is_real']
    
    # Re-identify categorical and numerical columns after normalization
    categorical_cols = []
    numerical_cols = []
    
    for col in X.columns:
        if X[col].dtype in ['object', 'category'] or X[col].nunique() <= 20:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Create preprocessing pipeline
    transformers = []
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    if numerical_cols:
        transformers.append(('num', 'passthrough', numerical_cols))
    
    if not transformers:
        raise ValueError("No valid columns for preprocessing")
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Create model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
    ])
    
    # Initialize KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Store predictions for all samples
    all_predictions = np.zeros(len(X))
    
    # Perform cross-validation
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model and get predictions
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]
        
        # Store predictions
        all_predictions[test_idx] = predictions
    
    # Calculate PMSE
    # We want predictions to be close to 0.5 (perfect indistinguishability)
    pmse = np.mean((all_predictions - 0.5) ** 2)
    
    return pmse
