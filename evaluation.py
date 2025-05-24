import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

class SyntheticDataEvaluator:
    """
    Comprehensive evaluation suite for synthetic data quality assessment
    """
    
    def __init__(self, real_data, synthetic_data, target_column='income'):
        """
        Initialize evaluator with real and synthetic datasets
        
        Args:
            real_data (pd.DataFrame): Original real dataset
            synthetic_data (pd.DataFrame): Generated synthetic dataset
            target_column (str): Name of the target column for ML evaluation
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.target_column = target_column
        
        # Ensure same columns
        common_columns = set(real_data.columns).intersection(set(synthetic_data.columns))
        self.real_data = self.real_data[list(common_columns)]
        self.synthetic_data = self.synthetic_data[list(common_columns)]
        
        self.results = {}
    
    def statistical_similarity(self):
        """
        Evaluate statistical similarity between real and synthetic data
        """
        print("=" * 60)
        print("STATISTICAL SIMILARITY EVALUATION")
        print("=" * 60)
        
        similarities = {}
        
        for column in self.real_data.columns:
            if self.real_data[column].dtype in ['int64', 'float64']:
                # Continuous variables
                similarities[column] = self._evaluate_continuous_column(column)
            else:
                # Categorical variables
                similarities[column] = self._evaluate_categorical_column(column)
        
        self.results['statistical_similarity'] = similarities
        
        # Summary statistics
        avg_similarity = np.mean([s['overall_score'] for s in similarities.values()])
        print(f"\nOverall Statistical Similarity Score: {avg_similarity:.3f}")
        print("(Higher is better, 1.0 = perfect match)")
        
        return similarities
    
    def _evaluate_continuous_column(self, column):
        """
        Evaluate continuous column similarity
        """
        real_values = self.real_data[column].dropna()
        synth_values = self.synthetic_data[column].dropna()
        
        # Basic statistics comparison
        real_stats = {
            'mean': real_values.mean(),
            'std': real_values.std(),
            'min': real_values.min(),
            'max': real_values.max(),
            'median': real_values.median()
        }
        
        synth_stats = {
            'mean': synth_values.mean(),
            'std': synth_values.std(),
            'min': synth_values.min(),
            'max': synth_values.max(),
            'median': synth_values.median()
        }
        
        # Calculate relative errors
        stat_errors = {}
        for stat_name in real_stats:
            if real_stats[stat_name] != 0:
                stat_errors[stat_name] = abs(real_stats[stat_name] - synth_stats[stat_name]) / abs(real_stats[stat_name])
            else:
                stat_errors[stat_name] = abs(synth_stats[stat_name])
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(real_values, synth_values)
        
        # Jensen-Shannon divergence (using histograms)
        try:
            # Create histograms
            bins = np.linspace(
                min(real_values.min(), synth_values.min()),
                max(real_values.max(), synth_values.max()),
                30
            )
            real_hist, _ = np.histogram(real_values, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_values, bins=bins, density=True)
            
            # Normalize to probabilities
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()
            
            js_divergence = jensenshannon(real_hist, synth_hist)
        except:
            js_divergence = 1.0
        
        # Overall score (lower KS statistic and JS divergence = better)
        overall_score = 1 - (ks_statistic + js_divergence) / 2
        
        print(f"{column} (continuous):")
        print(f"  KS Statistic: {ks_statistic:.3f} (lower is better)")
        print(f"  JS Divergence: {js_divergence:.3f} (lower is better)")
        print(f"  Overall Score: {overall_score:.3f}")
        
        return {
            'column_type': 'continuous',
            'real_stats': real_stats,
            'synth_stats': synth_stats,
            'stat_errors': stat_errors,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'js_divergence': js_divergence,
            'overall_score': max(0, overall_score)
        }
    
    def _evaluate_categorical_column(self, column):
        """
        Evaluate categorical column similarity
        """
        real_counts = self.real_data[column].value_counts(normalize=True).sort_index()
        synth_counts = self.synthetic_data[column].value_counts(normalize=True).sort_index()
        
        # Align indices
        all_categories = sorted(set(real_counts.index).union(set(synth_counts.index)))
        real_probs = [real_counts.get(cat, 0) for cat in all_categories]
        synth_probs = [synth_counts.get(cat, 0) for cat in all_categories]
        
        # Jensen-Shannon divergence
        js_divergence = jensenshannon(real_probs, synth_probs)
        
        # Chi-square test (if applicable)
        try:
            # Get absolute counts
            real_abs_counts = [self.real_data[column].value_counts().get(cat, 0) for cat in all_categories]
            synth_abs_counts = [self.synthetic_data[column].value_counts().get(cat, 0) for cat in all_categories]
            
            chi2_stat, chi2_pvalue = stats.chisquare(synth_abs_counts, real_abs_counts)
        except:
            chi2_stat, chi2_pvalue = float('inf'), 0
        
        # Total Variation Distance
        tv_distance = 0.5 * sum(abs(r - s) for r, s in zip(real_probs, synth_probs))
        
        # Overall score
        overall_score = 1 - js_divergence
        
        print(f"{column} (categorical):")
        print(f"  JS Divergence: {js_divergence:.3f} (lower is better)")
        print(f"  TV Distance: {tv_distance:.3f} (lower is better)")
        print(f"  Overall Score: {overall_score:.3f}")
        
        return {
            'column_type': 'categorical',
            'real_probs': dict(zip(all_categories, real_probs)),
            'synth_probs': dict(zip(all_categories, synth_probs)),
            'js_divergence': js_divergence,
            'tv_distance': tv_distance,
            'chi2_stat': chi2_stat,
            'chi2_pvalue': chi2_pvalue,
            'overall_score': max(0, overall_score)
        }
    
    def machine_learning_efficacy(self, test_size=0.2, random_state=42):
        """
        Evaluate machine learning efficacy using Train on Synthetic, Test on Real (TSTR)
        """
        print("\n" + "=" * 60)
        print("MACHINE LEARNING EFFICACY EVALUATION")
        print("=" * 60)
        
        # Prepare data
        X_real = self.real_data.drop(columns=[self.target_column])
        y_real = self.real_data[self.target_column]
        
        X_synth = self.synthetic_data.drop(columns=[self.target_column])
        y_synth = self.synthetic_data[self.target_column]
        
        # Split real data into train and test
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
        )
        
        # Encode categorical variables if needed
        categorical_columns = X_real.select_dtypes(include=['object']).columns
        
        # Prepare encoders
        encoders = {}
        for col in categorical_columns:
            encoders[col] = LabelEncoder()
            # Fit on combined real data to ensure consistent encoding
            all_values = list(X_real[col].unique()) + list(X_synth[col].unique())
            encoders[col].fit(all_values)
            
            X_real_train[col] = encoders[col].transform(X_real_train[col])
            X_real_test[col] = encoders[col].transform(X_real_test[col])
            X_synth[col] = encoders[col].transform(X_synth[col])
        
        # Encode target if needed
        if y_real.dtype == 'object':
            target_encoder = LabelEncoder()
            y_real_train_encoded = target_encoder.fit_transform(y_real_train)
            y_real_test_encoded = target_encoder.transform(y_real_test)
            y_synth_encoded = target_encoder.transform(y_synth)
        else:
            y_real_train_encoded = y_real_train
            y_real_test_encoded = y_real_test
            y_synth_encoded = y_synth
        
        # Models to evaluate
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(random_state=random_state, probability=True)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            
            # Train on Real, Test on Real (baseline)
            model_real = model.__class__(**model.get_params())
            model_real.fit(X_real_train, y_real_train_encoded)
            pred_real = model_real.predict(X_real_test)
            pred_proba_real = model_real.predict_proba(X_real_test)[:, 1] if hasattr(model_real, 'predict_proba') else None
            
            acc_real = accuracy_score(y_real_test_encoded, pred_real)
            f1_real = f1_score(y_real_test_encoded, pred_real, average='weighted')
            auc_real = roc_auc_score(y_real_test_encoded, pred_proba_real) if pred_proba_real is not None else None
            
            # Train on Synthetic, Test on Real
            model_synth = model.__class__(**model.get_params())
            model_synth.fit(X_synth, y_synth_encoded)
            pred_synth = model_synth.predict(X_real_test)
            pred_proba_synth = model_synth.predict_proba(X_real_test)[:, 1] if hasattr(model_synth, 'predict_proba') else None
            
            acc_synth = accuracy_score(y_real_test_encoded, pred_synth)
            f1_synth = f1_score(y_real_test_encoded, pred_synth, average='weighted')
            auc_synth = roc_auc_score(y_real_test_encoded, pred_proba_synth) if pred_proba_synth is not None else None
            
            # Calculate utility scores (how close synthetic performance is to real)
            acc_utility = acc_synth / acc_real if acc_real > 0 else 0
            f1_utility = f1_synth / f1_real if f1_real > 0 else 0
            auc_utility = auc_synth / auc_real if auc_real and auc_real > 0 else None
            
            # Format AUC values for display
            auc_real_str = f"{auc_real:.3f}" if auc_real is not None else "N/A"
            auc_synth_str = f"{auc_synth:.3f}" if auc_synth is not None else "N/A"
            auc_utility_str = f"{auc_utility:.3f}" if auc_utility is not None else "N/A"
            
            print(f"  Train on Real  → Accuracy: {acc_real:.3f}, F1: {f1_real:.3f}, AUC: {auc_real_str}")
            print(f"  Train on Synth → Accuracy: {acc_synth:.3f}, F1: {f1_synth:.3f}, AUC: {auc_synth_str}")
            print(f"  Utility Ratio  → Accuracy: {acc_utility:.3f}, F1: {f1_utility:.3f}, AUC: {auc_utility_str}")
            
            results[model_name] = {
                'train_real': {
                    'accuracy': acc_real,
                    'f1_score': f1_real,
                    'auc_score': auc_real
                },
                'train_synth': {
                    'accuracy': acc_synth,
                    'f1_score': f1_synth,
                    'auc_score': auc_synth
                },
                'utility_ratios': {
                    'accuracy': acc_utility,
                    'f1_score': f1_utility,
                    'auc_score': auc_utility
                }
            }
        
        # Overall ML efficacy score
        all_utility_scores = []
        for model_results in results.values():
            all_utility_scores.extend([
                model_results['utility_ratios']['accuracy'],
                model_results['utility_ratios']['f1_score']
            ])
            if model_results['utility_ratios']['auc_score'] is not None:
                all_utility_scores.append(model_results['utility_ratios']['auc_score'])
        
        overall_ml_score = np.mean(all_utility_scores)
        print(f"\nOverall ML Efficacy Score: {overall_ml_score:.3f}")
        print("(Higher is better, 1.0 = perfect utility preservation)")
        
        self.results['ml_efficacy'] = results
        self.results['overall_ml_score'] = overall_ml_score
        
        return results
    
    def privacy_assessment(self, n_samples=1000):
        """
        Basic privacy assessment using membership inference
        """
        print("\n" + "=" * 60)
        print("PRIVACY ASSESSMENT")
        print("=" * 60)
        
        # Simple membership inference attack
        # Sample from both datasets
        real_sample = self.real_data.sample(min(n_samples, len(self.real_data)), random_state=42)
        synth_sample = self.synthetic_data.sample(min(n_samples, len(self.synthetic_data)), random_state=42)
        
        # Calculate minimum distances between synthetic and real samples
        distances = []
        
        # Prepare numerical data for distance calculation
        real_numeric = real_sample.select_dtypes(include=[np.number])
        synth_numeric = synth_sample.select_dtypes(include=[np.number])
        
        if len(real_numeric.columns) > 0 and len(synth_numeric.columns) > 0:
            # Standardize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_numeric)
            synth_scaled = scaler.transform(synth_numeric)
            
            # Calculate minimum distances
            for synth_point in synth_scaled[:100]:  # Limit for computational efficiency
                dists = np.sqrt(np.sum((real_scaled - synth_point) ** 2, axis=1))
                distances.append(np.min(dists))
            
            avg_min_distance = np.mean(distances)
            privacy_score = min(1.0, avg_min_distance / np.std(distances)) if np.std(distances) > 0 else 1.0
        else:
            avg_min_distance = float('inf')
            privacy_score = 1.0
        
        print(f"Average minimum distance to real data: {avg_min_distance:.3f}")
        print(f"Privacy score: {privacy_score:.3f} (higher is better)")
        print("Note: This is a basic privacy assessment. Comprehensive privacy")
        print("evaluation requires more sophisticated techniques.")
        
        self.results['privacy_assessment'] = {
            'avg_min_distance': avg_min_distance,
            'privacy_score': privacy_score
        }
        
        return privacy_score
    
    def generate_report(self, save_path=None):
        """
        Generate comprehensive evaluation report
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)
        
        # Statistical similarity summary
        if 'statistical_similarity' in self.results:
            stat_scores = [s['overall_score'] for s in self.results['statistical_similarity'].values()]
            avg_stat_score = np.mean(stat_scores)
            print(f"Statistical Similarity Score: {avg_stat_score:.3f}/1.0")
        
        # ML efficacy summary
        if 'overall_ml_score' in self.results:
            print(f"ML Efficacy Score: {self.results['overall_ml_score']:.3f}/1.0")
        
        # Privacy summary
        if 'privacy_assessment' in self.results:
            print(f"Privacy Score: {self.results['privacy_assessment']['privacy_score']:.3f}/1.0")
        
        # Overall quality score
        scores = []
        if 'statistical_similarity' in self.results:
            scores.append(avg_stat_score)
        if 'overall_ml_score' in self.results:
            scores.append(self.results['overall_ml_score'])
        if 'privacy_assessment' in self.results:
            scores.append(self.results['privacy_assessment']['privacy_score'])
        
        if scores:
            overall_score = np.mean(scores)
            print(f"\nOverall Data Quality Score: {overall_score:.3f}/1.0")
            
            # Quality interpretation
            if overall_score >= 0.8:
                quality_level = "Excellent"
            elif overall_score >= 0.6:
                quality_level = "Good"
            elif overall_score >= 0.4:
                quality_level = "Fair"
            else:
                quality_level = "Poor"
            
            print(f"Quality Level: {quality_level}")
        
        print("=" * 60)
        
        if save_path:
            # Save detailed results to file
            with open(save_path, 'w') as f:
                f.write("Synthetic Data Quality Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                
                if 'statistical_similarity' in self.results:
                    f.write(f"Statistical Similarity Score: {avg_stat_score:.3f}/1.0\n")
                if 'overall_ml_score' in self.results:
                    f.write(f"ML Efficacy Score: {self.results['overall_ml_score']:.3f}/1.0\n")
                if 'privacy_assessment' in self.results:
                    f.write(f"Privacy Score: {self.results['privacy_assessment']['privacy_score']:.3f}/1.0\n")
                
                if scores:
                    f.write(f"\nOverall Data Quality Score: {overall_score:.3f}/1.0\n")
                    f.write(f"Quality Level: {quality_level}\n")
        
        return self.results
    
    def plot_distributions(self, columns=None, save_path=None):
        """
        Plot distribution comparisons between real and synthetic data
        """
        if columns is None:
            columns = list(self.real_data.columns)[:6]  # First 6 columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, column in enumerate(columns):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            if self.real_data[column].dtype in ['int64', 'float64']:
                # Continuous variables - histograms
                ax.hist(self.real_data[column].dropna(), alpha=0.6, label='Real', bins=30, density=True)
                ax.hist(self.synthetic_data[column].dropna(), alpha=0.6, label='Synthetic', bins=30, density=True)
            else:
                # Categorical variables - bar plots
                real_counts = self.real_data[column].value_counts(normalize=True)
                synth_counts = self.synthetic_data[column].value_counts(normalize=True)
                
                x = np.arange(len(real_counts))
                width = 0.35
                
                ax.bar(x - width/2, real_counts.values, width, alpha=0.6, label='Real')
                ax.bar(x + width/2, synth_counts.values, width, alpha=0.6, label='Synthetic')
                ax.set_xticks(x)
                ax.set_xticklabels(real_counts.index, rotation=45)
            
            ax.set_title(f'{column}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(columns), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 