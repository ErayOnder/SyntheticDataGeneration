import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import the preprocessing class
from data_preprocessor import AdultDataPreprocessor

class PrivBayesDP:
    def __init__(self, epsilon=1.0, delta=1e-5):
        """
        PrivBayes implementation for differentially private synthetic data generation
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter (for advanced composition if needed)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.network_structure = []
        self.noisy_distributions = {}
        self.feature_domains = {}
        self.feature_names = []
        
    def _add_laplace_noise(self, counts, epsilon_portion):
        """Add Laplace noise to counts for differential privacy"""
        if epsilon_portion <= 0:
            raise ValueError("Epsilon portion must be positive")
        
        scale = 1.0 / epsilon_portion
        noise = np.random.laplace(loc=0, scale=scale, size=counts.shape)
        noisy_counts = counts + noise
        
        # Ensure non-negative counts
        noisy_counts = np.clip(noisy_counts, 0, None)
        
        return noisy_counts
    
    def _discretize_continuous_features(self, df, continuous_features, n_bins=10):
        """Discretize continuous features into bins"""
        df_discretized = df.copy()
        
        for feature in continuous_features:
            if feature in df.columns:
                # Use quantile-based binning
                try:
                    df_discretized[feature] = pd.qcut(df[feature], 
                                                    q=n_bins, 
                                                    duplicates='drop',
                                                    labels=False)
                except ValueError:
                    # If qcut fails, use regular binning
                    df_discretized[feature] = pd.cut(df[feature], 
                                                   bins=n_bins, 
                                                   labels=False)
                
                # Handle any NaN values that might result from binning
                df_discretized[feature] = df_discretized[feature].fillna(0).astype(int)
        
        return df_discretized
    
    def _create_network_structure(self, features):
        """
        Create a simple Bayesian network structure
        For simplicity, we'll create a chain structure: feature1 -> feature2 -> ... -> featuren
        """
        structure = []
        
        # First feature has no parent (root)
        structure.append({'feature': features[0], 'parents': []})
        
        # Each subsequent feature depends on the previous one
        for i in range(1, len(features)):
            structure.append({'feature': features[i], 'parents': [features[i-1]]})
        
        return structure
    
    def fit(self, data, continuous_features=None, target_column=None):
        """
        Fit the PrivBayes model to the data
        
        Args:
            data: DataFrame containing the training data
            continuous_features: List of continuous feature names to discretize
            target_column: Name of target column (if any)
        """
        df = data.copy()
        
        # Discretize continuous features
        if continuous_features:
            df = self._discretize_continuous_features(df, continuous_features)
        
        # Store feature information
        self.feature_names = list(df.columns)
        
        # Create network structure
        self.network_structure = self._create_network_structure(self.feature_names)
        
        # Store feature domains (possible values for each feature)
        for feature in self.feature_names:
            self.feature_domains[feature] = sorted(df[feature].unique())
        
        # Split epsilon budget across all distributions
        n_distributions = len(self.network_structure)
        epsilon_per_distribution = self.epsilon / n_distributions
        
        print(f"Privacy budget allocation: ε = {self.epsilon}")
        print(f"Number of distributions: {n_distributions}")
        print(f"Epsilon per distribution: {epsilon_per_distribution:.3f}")
        
        # Learn noisy distributions
        for i, node in enumerate(self.network_structure):
            feature = node['feature']
            parents = node['parents']
            
            print(f"\nProcessing feature: {feature}")
            print(f"Parents: {parents if parents else 'None (root node)'}")
            
            if not parents:  # Root node - marginal distribution
                counts = df[feature].value_counts().sort_index()
                noisy_counts = self._add_laplace_noise(counts.values, epsilon_per_distribution)
                
                # Normalize to probabilities
                total = noisy_counts.sum()
                if total > 0:
                    probabilities = noisy_counts / total
                else:
                    # Uniform distribution if all counts are zero after noise
                    probabilities = np.ones(len(noisy_counts)) / len(noisy_counts)
                
                self.noisy_distributions[feature] = {
                    'type': 'marginal',
                    'probabilities': probabilities,
                    'domain': self.feature_domains[feature]
                }
                
            else:  # Conditional distribution
                parent = parents[0]  # We only handle one parent for simplicity
                
                # Create contingency table
                contingency = pd.crosstab(df[parent], df[feature], dropna=False)
                
                # Add Laplace noise to each cell
                noisy_contingency = self._add_laplace_noise(contingency.values, epsilon_per_distribution)
                
                # Normalize each row to get conditional probabilities
                conditional_probs = {}
                for idx, parent_value in enumerate(contingency.index):
                    row_sum = noisy_contingency[idx].sum()
                    if row_sum > 0:
                        conditional_probs[parent_value] = noisy_contingency[idx] / row_sum
                    else:
                        # Uniform distribution if all counts are zero
                        conditional_probs[parent_value] = np.ones(len(noisy_contingency[idx])) / len(noisy_contingency[idx])
                
                self.noisy_distributions[feature] = {
                    'type': 'conditional',
                    'parent': parent,
                    'conditional_probs': conditional_probs,
                    'domain': self.feature_domains[feature]
                }
        
        print(f"\nPrivBayes model fitted successfully!")
        print(f"Total privacy budget used: ε = {self.epsilon}")
        
    def generate_samples(self, n_samples):
        """
        Generate synthetic samples using the fitted PrivBayes model
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic samples
        """
        if not self.noisy_distributions:
            raise ValueError("Model must be fitted before generating samples")
        
        synthetic_data = []
        
        for _ in range(n_samples):
            sample = {}
            
            # Generate sample following the network structure
            for node in self.network_structure:
                feature = node['feature']
                parents = node['parents']
                
                if not parents:  # Root node
                    dist_info = self.noisy_distributions[feature]
                    probabilities = dist_info['probabilities']
                    domain = dist_info['domain']
                    
                    # Sample from marginal distribution
                    if len(probabilities) > 0 and probabilities.sum() > 0:
                        choice_idx = np.random.choice(len(domain), p=probabilities)
                        sample[feature] = domain[choice_idx]
                    else:
                        # Fallback to uniform sampling
                        sample[feature] = np.random.choice(domain)
                        
                else:  # Conditional node
                    parent = parents[0]
                    parent_value = sample[parent]
                    
                    dist_info = self.noisy_distributions[feature]
                    conditional_probs = dist_info['conditional_probs']
                    domain = dist_info['domain']
                    
                    if parent_value in conditional_probs:
                        probabilities = conditional_probs[parent_value]
                        if len(probabilities) > 0 and probabilities.sum() > 0:
                            choice_idx = np.random.choice(len(domain), p=probabilities)
                            sample[feature] = domain[choice_idx]
                        else:
                            sample[feature] = np.random.choice(domain)
                    else:
                        # If parent value not seen in training, use uniform
                        sample[feature] = np.random.choice(domain)
            
            synthetic_data.append(sample)
        
        return pd.DataFrame(synthetic_data)
    
    def get_privacy_cost(self):
        """Return the total privacy cost"""
        return self.epsilon, self.delta


def evaluate_synthetic_data(real_data, synthetic_data, feature_subset=None):
    """
    Basic evaluation of synthetic data quality
    """
    print("=== SYNTHETIC DATA EVALUATION ===")
    
    if feature_subset is None:
        feature_subset = real_data.columns[:5]  # First 5 features for quick evaluation
    
    print(f"\nEvaluating features: {list(feature_subset)}")
    
    # Statistical comparison
    print("\n--- Statistical Comparison ---")
    for feature in feature_subset:
        if feature in real_data.columns and feature in synthetic_data.columns:
            real_mean = real_data[feature].mean()
            synth_mean = synthetic_data[feature].mean()
            real_std = real_data[feature].std()
            synth_std = synthetic_data[feature].std()
            
            print(f"{feature}:")
            print(f"  Real:      Mean={real_mean:.3f}, Std={real_std:.3f}")
            print(f"  Synthetic: Mean={synth_mean:.3f}, Std={synth_std:.3f}")
            print(f"  Difference: Mean={abs(real_mean-synth_mean):.3f}, Std={abs(real_std-synth_std):.3f}")
    
    # Distribution comparison
    print("\n--- Distribution Comparison ---")
    for feature in feature_subset:
        if feature in real_data.columns and feature in synthetic_data.columns:
            real_dist = real_data[feature].value_counts(normalize=True).sort_index()
            synth_dist = synthetic_data[feature].value_counts(normalize=True).sort_index()
            
            print(f"\n{feature} - Top 3 values:")
            print("Real distribution:")
            print(real_dist.head(3))
            print("Synthetic distribution:")
            print(synth_dist.head(3))


def main():
    """
    Main function to demonstrate the PrivBayes implementation using AdultDataPreprocessor
    """
    print("=== WEEK 3: STAT-FIRST PIPELINE - PRIVBAYES with AdultDataPreprocessor ===")
    
    # Initialize the preprocessor
    print("\n1. Initializing data preprocessor...")
    preprocessor = AdultDataPreprocessor(data_dir='data')
    
    # Download data if not available
    print("2. Downloading data if needed...")
    try:
        preprocessor.download()
        print("Data download completed or files already exist")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Load the data
    print("3. Loading data...")
    try:
        df_train_orig, df_test_orig = preprocessor.load()
        print(f"Training data shape: {df_train_orig.shape}")
        print(f"Test data shape: {df_test_orig.shape}")
        print(f"Columns: {list(df_train_orig.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Combine and split data for our experiment
    print("4. Combining and splitting data...")
    try:
        df_train, df_test = preprocessor.combine_and_split(
            df_train_orig, df_test_orig, 
            test_size=0.2, 
            random_state=42
        )
        print(f"New training data shape: {df_train.shape}")
        print(f"New test data shape: {df_test.shape}")
    except Exception as e:
        print(f"Error combining and splitting data: {e}")
        return
    
    # Preprocess the data
    print("5. Preprocessing data...")
    try:
        df_train_proc, df_test_proc = preprocessor.preprocess(
            df_train, df_test,
            missing_strategy='fillna',  # Handle missing values by filling
            encoding='label',           # Use label encoding
            scale=False                 # Don't scale for PrivBayes (we'll discretize)
        )
        print(f"Processed training data shape: {df_train_proc.shape}")
        print(f"Processed test data shape: {df_test_proc.shape}")
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return
    
    # Identify continuous features that need discretization
    # These are the numerical features from the Adult dataset
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                          'capital-loss', 'hours-per-week']
    
    # Filter continuous features that actually exist in our processed data
    existing_continuous = [col for col in continuous_features if col in df_train_proc.columns]
    print(f"Continuous features to discretize: {existing_continuous}")
    
    # Test different privacy budgets
    privacy_budgets = [0.5, 1.0, 2.0]
    
    for epsilon in privacy_budgets:
        print(f"\n{'='*20} Testing with ε = {epsilon} {'='*20}")
        
        # Initialize PrivBayes
        privbayes = PrivBayesDP(epsilon=epsilon)
        
        # Fit the model
        print(f"\n6. Training PrivBayes model with ε = {epsilon}...")
        try:
            privbayes.fit(df_train_proc, continuous_features=existing_continuous)
        except Exception as e:
            print(f"Error fitting PrivBayes model: {e}")
            continue
        
        # Generate synthetic data
        print(f"\n7. Generating synthetic data...")
        try:
            n_synthetic = min(1000, len(df_train_proc))  # Generate up to 1000 samples
            synthetic_data = privbayes.generate_samples(n_synthetic)
            
            print(f"Generated {len(synthetic_data)} synthetic samples")
            print(f"Synthetic data shape: {synthetic_data.shape}")
            print("Sample of synthetic data:")
            print(synthetic_data.head())
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            continue
        
        # Basic evaluation
        print(f"\n8. Evaluating synthetic data quality...")
        try:
            # Select a subset of features for evaluation
            eval_features = df_train_proc.columns[:6]  # First 6 features
            evaluate_synthetic_data(df_train_proc, synthetic_data, eval_features)
        except Exception as e:
            print(f"Error evaluating synthetic data: {e}")
        
        # Privacy cost
        eps_used, delta_used = privbayes.get_privacy_cost()
        print(f"\nPrivacy cost: ε = {eps_used}, δ = {delta_used}")
        
        print("\n" + "="*80)
    
    print("\nPrivBayes implementation with AdultDataPreprocessor completed!")
    print("\nSummary:")
    print(f"- Successfully loaded and preprocessed {len(df_train_proc)} training samples")
    print(f"- Tested privacy budgets: {privacy_budgets}")
    print(f"- Generated synthetic data with differential privacy guarantees")
    print(f"- Used preprocessing pipeline from AdultDataPreprocessor")


if __name__ == "__main__":
    main()