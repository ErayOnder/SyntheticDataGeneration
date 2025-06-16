import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        save_path (Optional[str]): Path to save the plot. If None, plot is displayed
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
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

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
        save_path (Optional[str]): Path to save the plot. If None, plot is displayed
        
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
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    
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
    # Create copies to avoid modifying original dataframes
    real_df = real_df.copy()
    synth_df = synth_df.copy()
    
    # Add source indicator
    real_df['is_real'] = 1
    synth_df['is_real'] = 0
    
    # Combine datasets
    combined_df = pd.concat([real_df, synth_df], axis=0, ignore_index=True)
    
    # Separate features and target
    X = combined_df.drop('is_real', axis=1)
    y = combined_df['is_real']
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    
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
