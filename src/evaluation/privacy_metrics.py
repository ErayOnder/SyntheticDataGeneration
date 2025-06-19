import pandas as pd
import numpy as np

def count_exact_matches(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame
) -> int:
    """
    Count the number of rows in synthetic data that exactly match rows in real data.
    
    This metric helps assess privacy risks in synthetic data generation.
    A high number of exact matches indicates potential privacy concerns.
    
    Args:
        real_df (pd.DataFrame): DataFrame containing real data
        synth_df (pd.DataFrame): DataFrame containing synthetic data
        
    Returns:
        int: Number of exact matches found
    """
    # Create copies to avoid modifying original dataframes
    real_df = real_df.copy()
    synth_df = synth_df.copy()
    
    # Convert all columns to string type to ensure consistent comparison
    real_df = real_df.astype(str)
    synth_df = synth_df.astype(str)
    
    # Create sets of row tuples for efficient comparison
    real_rows = set(tuple(row) for row in real_df.values)
    synth_rows = set(tuple(row) for row in synth_df.values)
    
    # Count exact matches
    exact_matches = len(real_rows.intersection(synth_rows))
    
    return exact_matches

def calculate_dcr(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame
) -> dict:
    """
    Calculate Distance to Closest Record (DCR) metrics using Gower distance.
    
    DCR measures how close synthetic records are to real records.
    Higher distances indicate better privacy protection.
    
    Args:
        synth_df (pd.DataFrame): DataFrame containing synthetic data
        real_df (pd.DataFrame): DataFrame containing real data
        
    Returns:
        dict: Dictionary containing mean, min, and max of minimum distances
    """
    try:
        import gower
    except ImportError:
        raise ImportError(
            "The gower package is required. Please install it using: pip install gower"
        )
    
    # Create copies to avoid modifying original dataframes
    synth_df = synth_df.copy()
    real_df = real_df.copy()
    
    # Calculate Gower distance matrix between synthetic and real data
    distance_matrix = gower.gower_matrix(synth_df, real_df)
    
    # For each synthetic record, find the minimum distance to any real record
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate statistics
    dcr_stats = {
        'mean': float(np.mean(min_distances)),
        'min': float(np.min(min_distances)),
        'max': float(np.max(min_distances))
    }
    
    return dcr_stats
