import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessor import AdultDataPreprocessor
from privbayes import MyPrivBayes

def plot_marginal_distributions(real_df, synth_dfs, column, epsilons, output_dir='epsilon_plots'):
    """
    Plot marginal distributions for different epsilon values.
    
    Args:
        real_df (pd.DataFrame): Real dataset
        synth_dfs (dict): Dictionary of synthetic datasets for different epsilon values
        column (str): Column to plot
        epsilons (list): List of epsilon values used
        output_dir (str): Directory to save the plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate distributions
    real_dist = real_df[column].value_counts(normalize=True).sort_index()
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot real distribution
    plt.bar(real_dist.index, real_dist.values, alpha=0.3, label='Real Data', color='gray')
    
    # Plot synthetic distributions
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))
    for eps, color in zip(epsilons, colors):
        synth_dist = synth_dfs[eps][column].value_counts(normalize=True).sort_index()
        plt.plot(synth_dist.index, synth_dist.values, 'o-', label=f'ε={eps}', color=color, alpha=0.7)
    
    # Customize plot
    plt.title(f'Marginal Distribution of {column} Across Different ε Values', fontsize=14, pad=20)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'marginal_{column}_epsilon_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_epsilon_heatmap(real_df, synth_dfs, columns, epsilons, output_dir='epsilon_plots'):
    """
    Create a heatmap showing the difference between real and synthetic distributions
    for different epsilon values.
    
    Args:
        real_df (pd.DataFrame): Real dataset
        synth_dfs (dict): Dictionary of synthetic datasets for different epsilon values
        columns (list): List of columns to analyze
        epsilons (list): List of epsilon values used
        output_dir (str): Directory to save the plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate differences for each column and epsilon
    differences = []
    for col in columns:
        real_dist = real_df[col].value_counts(normalize=True).sort_index()
        for eps in epsilons:
            synth_dist = synth_dfs[eps][col].value_counts(normalize=True).sort_index()
            # Calculate total variation distance
            tvd = 0.5 * sum(abs(real_dist.get(cat, 0) - synth_dist.get(cat, 0)) 
                           for cat in set(real_dist.index) | set(synth_dist.index))
            differences.append({
                'Column': col,
                'Epsilon': eps,
                'TVD': tvd
            })
    
    # Create DataFrame for heatmap
    diff_df = pd.DataFrame(differences)
    heatmap_data = diff_df.pivot(index='Column', columns='Epsilon', values='TVD')
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Total Variation Distance Between Real and Synthetic Distributions', pad=20)
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Column')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epsilon_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🚀 Starting Epsilon Evaluation")
    print("=" * 50)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    preprocessor = AdultDataPreprocessor()
    preprocessor.download()
    df_train, df_test = preprocessor.load()
    df_real = preprocessor.preprocess_for_privbayes(pd.concat([df_train, df_test], ignore_index=True))
    
    print(f"✓ Real data shape: {df_real.shape}")
    
    # Define epsilon values to test
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"\n🔒 Testing epsilon values: {epsilons}")
    
    # Generate synthetic data for each epsilon
    synth_dfs = {}
    for eps in epsilons:
        print(f"\n🧠 Training MyPrivBayes with ε={eps}...")
        synth = MyPrivBayes(epsilon=eps)
        synth.fit(df_real)
        synth_dfs[eps] = synth.sample(len(df_real))
        print(f"✓ Generated synthetic data for ε={eps}")
    
    # Define columns to analyze
    columns_to_analyze = ['income', 'education', 'workclass', 'race']
    print(f"\n📈 Analyzing distributions for: {columns_to_analyze}")
    
    # Generate plots for each column
    print("\n🎨 Generating marginal distribution plots...")
    for column in columns_to_analyze:
        plot_marginal_distributions(df_real, synth_dfs, column, epsilons)
        print(f"✓ Generated plot for {column}")
    
    # Generate heatmap
    print("\n🎨 Generating epsilon heatmap...")
    plot_epsilon_heatmap(df_real, synth_dfs, columns_to_analyze, epsilons)
    print("✓ Generated epsilon heatmap")
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("📁 Check the 'epsilon_plots' directory for visualization files")
    print("=" * 60) 