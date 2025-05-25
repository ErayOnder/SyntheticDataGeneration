import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessor import AdultDataPreprocessor
from privbayes import MyPrivBayes

def plot_comparison(real_df, synth_df, columns_to_plot, output_dir='comparison_plots'):
    """
    Create side-by-side bar plots comparing distributions across real and synthetic datasets.
    
    Args:
        real_df (pd.DataFrame): Real dataset
        synth_df (pd.DataFrame): Synthetic dataset (MyPrivBayes)
        columns_to_plot (list): List of column names to plot
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create copies to avoid modifying original DataFrames
    real_copy = real_df.copy()
    synth_copy = synth_df.copy()
    
    # Add source column to each DataFrame
    real_copy['Source'] = 'Real Data'
    synth_copy['Source'] = 'MyPrivBayes'
    
    # Combine DataFrames
    combined_df = pd.concat([real_copy, synth_copy], ignore_index=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots for each column
    for column in columns_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Create count plot
        ax = sns.countplot(data=combined_df, x=column, hue='Source')
        
        # Customize plot
        plt.title(f'Distribution Comparison: {column}', fontsize=16, pad=20)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        plt.legend(title='Dataset', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{column}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison plot for {column}")

if __name__ == "__main__":
    print("🚀 Starting PrivBayes Evaluation")
    print("=" * 50)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    preprocessor = AdultDataPreprocessor()
    preprocessor.download()
    df_train, df_test = preprocessor.load()
    df_real = preprocessor.preprocess_for_privbayes(pd.concat([df_train, df_test], ignore_index=True))
    
    print(f"✓ Real data shape: {df_real.shape}")
    print(f"✓ Columns: {list(df_real.columns)}")
    
    # Set privacy budget
    epsilon = 5.0
    print(f"🔒 Privacy budget (epsilon): {epsilon}")
    
    # Train and generate synthetic data
    print("\n🧠 Training MyPrivBayes and generating synthetic data...")
    synth = MyPrivBayes(epsilon=epsilon)
    synth.fit(df_real)
    df_synth = synth.sample(len(df_real))
    
    print(f"✓ Synthetic data shape: {df_synth.shape}")
    
    # Define columns to compare
    columns_to_compare = ['income', 'education', 'workclass', 'race']
    print(f"\n📈 Comparing distributions for: {columns_to_compare}")
    
    # Generate comparison plots
    print("\n🎨 Generating comparison plots...")
    plot_comparison(df_real, df_synth, columns_to_compare)
    
    print("\n📊 Generating summary statistics...")
    
    # Print basic statistics comparison
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPARISON SUMMARY")
    print("=" * 60)
    
    for col in columns_to_compare:
        print(f"\n📋 {col.upper()}:")
        print("-" * 40)
        
        real_dist = df_real[col].value_counts().sort_index()
        synth_dist = df_synth[col].value_counts().sort_index()
        
        # Ensure both have same categories
        all_categories = sorted(set(real_dist.index) | set(synth_dist.index))
        
        comparison_df = pd.DataFrame({
            'Real_Count': [real_dist.get(cat, 0) for cat in all_categories],
            'Synthetic_Count': [synth_dist.get(cat, 0) for cat in all_categories],
        }, index=all_categories)
        
        comparison_df['Real_Pct'] = (comparison_df['Real_Count'] / comparison_df['Real_Count'].sum() * 100).round(1)
        comparison_df['Synthetic_Pct'] = (comparison_df['Synthetic_Count'] / comparison_df['Synthetic_Count'].sum() * 100).round(1)
        comparison_df['Difference'] = (comparison_df['Synthetic_Pct'] - comparison_df['Real_Pct']).round(1)
        
        print(comparison_df)
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("📁 Check the 'comparison_plots' directory for visualization files")
    print("=" * 60) 