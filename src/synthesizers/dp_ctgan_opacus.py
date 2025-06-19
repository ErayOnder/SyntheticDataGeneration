import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Opacus imports for differential privacy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants.utils import get_noise_multiplier

# Note: We implement our own CTGAN components below with proper categorical handling


class DataTransformer:
    """
    CTGAN-style data transformer with mode-specific normalization
    """
    
    def __init__(self):
        self.output_info = []
        self.output_dimensions = 0
        self.dataframe = True
        self.column_names = []
        
    def fit(self, raw_data, discrete_columns=None):
        """
        Fit the data transformer
        """
        if discrete_columns is None:
            discrete_columns = []
            
        self.column_names = raw_data.columns.tolist()
        self.output_info = []
        self.output_dimensions = 0
        
        # Store original discrete columns
        self.discrete_columns = discrete_columns
        
        for column in raw_data.columns:
            if column in discrete_columns:
                # Discrete column processing
                column_data = raw_data[column].astype(str)
                categories = column_data.value_counts().index.tolist()
                
                # Store discrete column info
                self.output_info.append({
                    'name': column,
                    'type': 'discrete',
                    'size': len(categories),
                    'categories': categories
                })
                self.output_dimensions += len(categories)
                
            else:
                # Continuous column processing with mode-specific normalization
                column_data = pd.to_numeric(raw_data[column], errors='coerce').dropna()
                
                if len(column_data) == 0:
                    continue
                    
                # Use Bayesian Gaussian Mixture for mode detection
                gm = BayesianGaussianMixture(
                    n_components=min(10, len(column_data) // 50 + 1),
                    max_iter=100,
                    random_state=42,
                    tol=1e-3
                )
                
                # Reshape for sklearn
                column_data_reshaped = column_data.values.reshape(-1, 1)
                gm.fit(column_data_reshaped)
                
                # Get valid components (with non-zero weights)
                valid_components = gm.weights_ > 0.01
                n_modes = np.sum(valid_components)
                
                # If no valid modes found, use single mode
                if n_modes == 0:
                    n_modes = 1
                    means = np.array([column_data.mean()])
                    stds = np.array([max(column_data.std(), 1e-6)])
                    weights = np.array([1.0])
                else:
                    # Extract mode parameters
                    means = gm.means_[valid_components].flatten()
                    stds = np.sqrt(gm.covariances_[valid_components]).flatten()
                    weights = gm.weights_[valid_components]
                    
                    # Ensure minimum std to avoid division by zero
                    stds = np.maximum(stds, 1e-6)
                    
                    # Normalize weights
                    weights = weights / np.sum(weights)
                
                # Store continuous column info
                self.output_info.append({
                    'name': column,
                    'type': 'continuous',
                    'modes': n_modes,
                    'means': means,
                    'stds': stds,
                    'weights': weights,
                    'gm': gm
                })
                
                # Add dimensions: 1 scalar + n_modes indicator
                self.output_dimensions += 1 + n_modes
    
    def transform(self, raw_data):
        """
        Transform the data using mode-specific normalization
        """
        transformed_data = []
        
        for i, info in enumerate(self.output_info):
            column_name = info['name']
            
            if info['type'] == 'discrete':
                # One-hot encode discrete columns
                column_data = raw_data[column_name].astype(str)
                for category in info['categories']:
                    transformed_data.append((column_data == category).astype(float))
                    
            else:  # continuous
                column_data = pd.to_numeric(raw_data[column_name], errors='coerce')
                
                # Handle missing values
                column_data = column_data.fillna(column_data.mean())
                
                # Mode-specific normalization
                n_modes = info['modes']
                means = info['means']
                stds = info['stds']
                weights = info['weights']
                
                # Calculate probabilities for each mode
                probs = np.zeros((len(column_data), n_modes))
                for j in range(n_modes):
                    probs[:, j] = weights[j] * self._gaussian_pdf(column_data, means[j], stds[j])
                
                # Normalize probabilities
                probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
                
                # Sample mode for each value based on probabilities
                modes = []
                scalars = []
                for k in range(len(column_data)):
                    prob_k = probs[k]
                    prob_sum = np.sum(prob_k)
                    
                    if prob_sum == 0 or np.isnan(prob_sum):
                        # If all probabilities are zero or nan, choose mode randomly
                        selected_mode = np.random.choice(n_modes)
                    else:
                        # Normalize probabilities to sum to 1
                        prob_k = prob_k / prob_sum
                        selected_mode = np.random.choice(n_modes, p=prob_k)
                    
                    modes.append(selected_mode)
                    
                    # Normalize value within selected mode
                    scalar = (column_data.iloc[k] - means[selected_mode]) / (4 * stds[selected_mode])
                    scalar = np.clip(scalar, -1, 1)
                    scalars.append(scalar)
                
                # Add scalar values
                transformed_data.append(np.array(scalars))
                
                # Add mode indicators (one-hot)
                for j in range(n_modes):
                    mode_indicator = np.array([1.0 if mode == j else 0.0 for mode in modes])
                    transformed_data.append(mode_indicator)
        
        # Stack all features
        result = np.column_stack(transformed_data)
        return result
    
    def inverse_transform(self, data):
        """
        Inverse transform the data back to original space
        """
        recovered_data = []
        column_names = []
        data_ptr = 0
        
        for info in self.output_info:
            column_names.append(info['name'])
            
            if info['type'] == 'discrete':
                # Recover discrete columns
                n_categories = info['size']
                column_data = data[:, data_ptr:data_ptr + n_categories]
                
                # Find the category with highest probability
                category_indices = np.argmax(column_data, axis=1)
                recovered_column = [info['categories'][idx] for idx in category_indices]
                recovered_data.append(recovered_column)
                
                data_ptr += n_categories
                
            else:  # continuous
                n_modes = info['modes']
                means = info['means']
                stds = info['stds']
                
                # Get scalar values
                scalars = data[:, data_ptr]
                data_ptr += 1
                
                # Get mode indicators
                mode_indicators = data[:, data_ptr:data_ptr + n_modes]
                data_ptr += n_modes
                
                # Find selected modes
                selected_modes = np.argmax(mode_indicators, axis=1)
                
                # Recover original values
                recovered_column = []
                for i, (scalar, mode_idx) in enumerate(zip(scalars, selected_modes)):
                    # Denormalize using selected mode parameters
                    original_value = scalar * (4 * stds[mode_idx]) + means[mode_idx]
                    recovered_column.append(original_value)
                
                recovered_data.append(recovered_column)
        
        # Create DataFrame
        recovered_df = pd.DataFrame(
            np.column_stack(recovered_data),
            columns=column_names
        )
        
        # Convert columns back to proper types
        for info in self.output_info:
            if info['type'] == 'discrete':
                recovered_df[info['name']] = recovered_df[info['name']].astype(str)
            else:
                # Convert continuous columns to numeric
                recovered_df[info['name']] = pd.to_numeric(recovered_df[info['name']], errors='coerce')
        
        return recovered_df
    
    def _gaussian_pdf(self, x, mean, std):
        """Calculate Gaussian PDF"""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


class DataSampler:
    """
    CTGAN-style data sampler for conditional training
    """
    
    def __init__(self, data, output_info, log_frequency=True):
        self.data = data
        self.output_info = output_info
        self.log_frequency = log_frequency
        
        # Find discrete columns and their positions
        self.discrete_columns = []
        self.discrete_column_matrix_st = []
        
        st = 0
        for info in output_info:
            if info['type'] == 'discrete':
                self.discrete_columns.append(info)
                self.discrete_column_matrix_st.append(st)
            st += info.get('size', 1 + info.get('modes', 0))
        
        # Calculate conditional probabilities for training-by-sampling
        self.conditional_probs = []
        for i, (info, st) in enumerate(zip(self.discrete_columns, self.discrete_column_matrix_st)):
            # Get the discrete column data
            column_data = data[:, st:st + info['size']]
            category_counts = np.sum(column_data, axis=0)
            
            if self.log_frequency:
                # Use log frequency for balanced sampling
                probs = np.log(category_counts + 1)
            else:
                probs = category_counts
            
            probs = probs / np.sum(probs)
            self.conditional_probs.append(probs)
    
    def sample_condvec(self, batch_size):
        """
        Sample conditional vector for training
        """
        if len(self.discrete_columns) == 0:
            return None
        
        # Randomly select a discrete column
        col_idx = np.random.choice(len(self.discrete_columns))
        selected_column = self.discrete_columns[col_idx]
        
        # Sample categories according to conditional probabilities  
        category_indices = np.random.choice(
            selected_column['size'],
            size=batch_size,
            p=self.conditional_probs[col_idx]
        )
        
        # Create conditional vector
        cond_vec = np.zeros((batch_size, self.dim_cond_vec()))
        
        # Fill in the conditional vector
        st = 0
        for i, info in enumerate(self.discrete_columns):
            if i == col_idx:
                for j, cat_idx in enumerate(category_indices):
                    cond_vec[j, st + cat_idx] = 1
            st += info['size']
        
        # Return: conditional vector, mask, column index, category indices
        return cond_vec, None, col_idx, category_indices
    
    def dim_cond_vec(self):
        """Get dimension of conditional vector"""
        return sum(info['size'] for info in self.discrete_columns)


def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    """Gumbel softmax activation"""
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        # Straight through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    
    return ret


class Generator(nn.Module):
    """
    CTGAN Generator with conditional input
    """
    
    def __init__(self, latent_dim, data_dim, cond_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        
        input_dim = latent_dim + cond_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.main = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, data_dim)
        
    def forward(self, noise, cond=None):
        if cond is not None:
            input_tensor = torch.cat([noise, cond], dim=1)
        else:
            input_tensor = noise
            
        hidden = self.main(input_tensor)
        output = self.output_layer(hidden)
        
        return output


class Discriminator(nn.Module):
    """
    CTGAN Discriminator (Critic) with conditional input
    """
    
    def __init__(self, data_dim, cond_dim, pac_size=10, hidden_dims=[256, 256]):
        super().__init__()
        
        self.pac_size = pac_size
        input_dim = (data_dim + cond_dim) * pac_size
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.main = nn.Sequential(*layers)
        
    def forward(self, data, cond=None):
        if cond is not None:
            input_tensor = torch.cat([data, cond], dim=1)
        else:
            input_tensor = data
            
        return self.main(input_tensor)


class OpacusDifferentiallyPrivateCTGAN:
    """
    Improved DP-CTGAN with proper categorical data handling based on the original CTGAN paper
    
    Key improvements:
    1. Mode-specific normalization for continuous columns
    2. Conditional generator and training-by-sampling for categorical columns
    3. Proper Gumbel softmax for discrete outputs
    4. PacGAN integration for mode collapse prevention
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, max_grad_norm=1.0, 
                 epochs=100, batch_size=500, generator_lr=2e-4, discriminator_lr=2e-4,
                 generator_decay=1e-6, discriminator_decay=1e-6, discriminator_steps=1,
                 pac_size=10, tau=0.2, verbose=True):
        """
        Initialize improved DP-CTGAN
        
        Args:
            epsilon (float): Privacy budget parameter
            delta (float): Relaxation parameter for (ε,δ)-DP
            max_grad_norm (float): Maximum gradient norm for clipping
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            generator_lr (float): Generator learning rate
            discriminator_lr (float): Discriminator learning rate
            generator_decay (float): Generator weight decay
            discriminator_decay (float): Discriminator weight decay
            discriminator_steps (int): Discriminator steps per generator step
            pac_size (int): PacGAN pac size
            tau (float): Gumbel softmax temperature
            verbose (bool): Print training progress
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.generator_decay = generator_decay
        self.discriminator_decay = discriminator_decay
        self.discriminator_steps = discriminator_steps
        self.pac_size = pac_size
        self.tau = tau
        self.verbose = verbose
        
        # Model components
        self.generator = None
        self.discriminator = None
        self.data_transformer = None
        self.data_sampler = None
        
        # Privacy tracking
        self.privacy_engine = None
        self.noise_multiplier = None
        self.privacy_spent = 0.0
        
        # Training history
        self.training_losses = {'generator': [], 'discriminator': []}
        self.privacy_history = []
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print(f"Using device: {self.device}")
            print(f"Privacy parameters: ε={epsilon}, δ={delta}")
    
    def _prepare_data(self, data, discrete_columns):
        """
        Prepare data using CTGAN-style transformation with pre-identified discrete columns.
        
        This method now expects discrete_columns to be provided by the unified preprocessing system
        instead of auto-detecting them, ensuring consistency across the evaluation pipeline.
        
        Args:
            data (pd.DataFrame): Preprocessed data from unified preprocessing system
            discrete_columns (list): List of discrete column names from preprocessing metadata
        """
        if self.verbose:
            print(f"Using provided discrete columns: {discrete_columns}")
            print(f"Continuous columns: {[c for c in data.columns if c not in discrete_columns]}")
        
        # Initialize and fit data transformer with provided discrete columns
        self.data_transformer = DataTransformer()
        self.data_transformer.fit(data, discrete_columns)
        
        # Transform data
        transformed_data = self.data_transformer.transform(data)
        
        if self.verbose:
            print(f"Transformed data shape: {transformed_data.shape}")
            print(f"Original columns: {len(data.columns)}, Transformed features: {transformed_data.shape[1]}")
        
        # Create data sampler
        self.data_sampler = DataSampler(
            transformed_data,
            self.data_transformer.output_info,
            log_frequency=True
        )
        
        return transformed_data
    
    def _compute_privacy_parameters(self, dataset_size):
        """Compute privacy parameters using Opacus utilities"""
        sample_rate = self.batch_size / dataset_size
        
        # Calculate noise multiplier for given privacy budget
        self.noise_multiplier = get_noise_multiplier(
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            sample_rate=sample_rate,
            epochs=self.epochs
        )
        
        if self.verbose:
            print(f"Sample rate: {sample_rate:.4f}")
            print(f"Noise multiplier: {self.noise_multiplier:.4f}")
        
        return self.noise_multiplier
    
    def _create_models(self, data_dim, cond_dim):
        """Create generator and discriminator models"""
        # Generator
        self.generator = Generator(
            latent_dim=128,
            data_dim=data_dim,
            cond_dim=cond_dim
        ).to(self.device)
        
        # Discriminator with PacGAN
        self.discriminator = Discriminator(
            data_dim=data_dim,
            cond_dim=cond_dim,
            pac_size=self.pac_size
        ).to(self.device)
        
        if self.verbose:
            print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
            print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _apply_activate(self, data):
        """
        Apply appropriate activation functions to generated data
        """
        activated_data = []
        data_ptr = 0
        
        for info in self.data_transformer.output_info:
            if info['type'] == 'discrete':
                # Apply Gumbel softmax to discrete columns
                n_categories = info['size']
                logits = data[:, data_ptr:data_ptr + n_categories]
                activated = gumbel_softmax(logits, tau=self.tau, hard=False)
                activated_data.append(activated)
                data_ptr += n_categories
                
            else:  # continuous
                n_modes = info['modes']
                
                # Scalar value with tanh activation
                scalar = torch.tanh(data[:, data_ptr:data_ptr + 1])
                activated_data.append(scalar)
                data_ptr += 1
                
                # Mode indicator with Gumbel softmax
                logits = data[:, data_ptr:data_ptr + n_modes]
                mode_indicator = gumbel_softmax(logits, tau=self.tau, hard=False)
                activated_data.append(mode_indicator)
                data_ptr += n_modes
        
        return torch.cat(activated_data, dim=1)
    
    def _create_dataloader(self, transformed_data):
        """Create PyTorch DataLoader for training"""
        data_tensor = torch.FloatTensor(transformed_data).to(self.device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        return dataloader
    
    def _sample_condvec(self, batch_size):
        """Sample conditional vector for conditional training"""
        condvec_result = self.data_sampler.sample_condvec(batch_size)
        if condvec_result is None:
            return None, None, None, None
        
        cond_vec, mask, col_idx, cat_indices = condvec_result
        cond_vec = torch.FloatTensor(cond_vec).to(self.device)
        
        return cond_vec, mask, col_idx, cat_indices
    
    def fit(self, data, discrete_columns=None):
        """
        Train the improved DP-CTGAN
        
        Args:
            data (pd.DataFrame): Preprocessed data from unified preprocessing system
            discrete_columns (list): List of discrete column names from preprocessing metadata.
                                    If None, will auto-detect (for backward compatibility)
        """
        print("=" * 60)
        print("TRAINING IMPROVED DP-CTGAN WITH PROPER CATEGORICAL HANDLING")
        print("=" * 60)
        print(f"Privacy Parameters: ε={self.epsilon}, δ={self.delta}")
        print(f"Max Gradient Norm: {self.max_grad_norm}")
        print(f"Training Configuration: {self.epochs} epochs, batch size {self.batch_size}")
        print(f"PacGAN size: {self.pac_size}, Gumbel tau: {self.tau}")
        print("=" * 60)
        
        # Handle discrete columns (backward compatibility)
        if discrete_columns is None:
            print("-> No discrete_columns provided - auto-detecting for backward compatibility")
            discrete_columns = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    discrete_columns.append(col)
                elif data[col].nunique() <= 10:
                    discrete_columns.append(col)
        
        # Prepare data with provided discrete columns
        transformed_data = self._prepare_data(data, discrete_columns)
        dataset_size = len(data)
        
        # Compute privacy parameters
        self._compute_privacy_parameters(dataset_size)
        
        # Get dimensions
        data_dim = transformed_data.shape[1]
        cond_dim = self.data_sampler.dim_cond_vec() if self.data_sampler.dim_cond_vec() > 0 else 0
        
        if self.verbose:
            print(f"Data dimension: {data_dim}")
            print(f"Conditional dimension: {cond_dim}")
        
        # Create models
        self._create_models(data_dim, cond_dim)
        
        # Create optimizers
        optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self.generator_decay
        )
        
        optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self.discriminator_decay
        )
        
        # Create dataloader
        train_loader = self._create_dataloader(transformed_data)
        
        # Setup Opacus Privacy Engine for discriminator only
        self.privacy_engine = PrivacyEngine()
        
        # Make the discriminator private
        self.discriminator, optimizer_d, train_loader = self.privacy_engine.make_private(
            module=self.discriminator,
            optimizer=optimizer_d,
            data_loader=train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=False
        )
        
        if self.verbose:
            print(f"Privacy engine initialized")
        
        # Training loop
        self._train_loop(optimizer_g, optimizer_d, train_loader, transformed_data)
        
        print("Training completed!")
        print("=" * 60)
    
    def _train_loop(self, optimizer_g, optimizer_d, train_loader, transformed_data):
        """Main training loop with improved CTGAN techniques"""
        
        for epoch in tqdm(range(self.epochs), desc="Training Improved DP-CTGAN"):
            g_losses = []
            d_losses = []
            
            for batch_idx, (real_data,) in enumerate(train_loader):
                batch_size = real_data.size(0)
                
                # Sample conditional vector for this batch
                cond_vec, mask, col_idx, cat_indices = self._sample_condvec(batch_size)
                
                # Train Discriminator
                for _ in range(self.discriminator_steps):
                    optimizer_d.zero_grad()
                    
                    # Generate fake data
                    noise = torch.randn(batch_size, 128).to(self.device)
                    with torch.no_grad():
                        fake_raw = self.generator(noise, cond_vec)
                        fake_data = self._apply_activate(fake_raw)
                    
                    # Create PacGAN batches
                    if self.pac_size > 1:
                        # Reshape for PacGAN
                        real_pac = self._create_pac_batch(real_data, cond_vec)
                        fake_pac = self._create_pac_batch(fake_data, cond_vec)
                    else:
                        real_pac = torch.cat([real_data, cond_vec], dim=1) if cond_vec is not None else real_data
                        fake_pac = torch.cat([fake_data, cond_vec], dim=1) if cond_vec is not None else fake_data
                    
                    # Discriminator predictions
                    real_pred = self.discriminator(real_pac)
                    fake_pred = self.discriminator(fake_pac)
                    
                    # Simple Wasserstein loss (no gradient penalty with Opacus due to compatibility issues)
                    d_loss = torch.mean(fake_pred) - torch.mean(real_pred)
                    
                    d_loss.backward()
                    optimizer_d.step()
                    
                    # Weight clipping for WGAN stability (instead of gradient penalty with Opacus)
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    
                    d_losses.append(d_loss.item())
                
                # Train Generator
                optimizer_g.zero_grad()
                
                # Generate fake data
                noise = torch.randn(batch_size, 128).to(self.device)
                fake_raw = self.generator(noise, cond_vec)
                fake_data = self._apply_activate(fake_raw)
                
                # Create PacGAN batch for generator
                if self.pac_size > 1:
                    fake_pac = self._create_pac_batch(fake_data, cond_vec)
                else:
                    fake_pac = torch.cat([fake_data, cond_vec], dim=1) if cond_vec is not None else fake_data
                
                # Generator loss
                fake_pred = self.discriminator(fake_pac)
                g_loss = -torch.mean(fake_pred)
                
                # Add conditional loss if we have discrete columns
                if cond_vec is not None and col_idx is not None:
                    # Calculate cross-entropy loss for the conditional column
                    cond_loss = self._calculate_conditional_loss(fake_data, cond_vec, col_idx)
                    g_loss += cond_loss
                
                g_loss.backward()
                optimizer_g.step()
                
                g_losses.append(g_loss.item())
            
            # Store losses
            self.training_losses['generator'].append(np.mean(g_losses))
            self.training_losses['discriminator'].append(np.mean(d_losses))
            
            # Get privacy spent from Opacus
            epsilon_spent = self.privacy_engine.get_epsilon(self.delta)
            self.privacy_history.append(epsilon_spent)
            
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"G_loss: {np.mean(g_losses):.4f}, "
                      f"D_loss: {np.mean(d_losses):.4f}, "
                      f"Privacy spent: ε={epsilon_spent:.4f}")
        
        # Final privacy accounting
        self.privacy_spent = self.privacy_engine.get_epsilon(self.delta)
    
    def _create_pac_batch(self, data, cond_vec):
        """Create PacGAN batches"""
        batch_size = data.size(0)
        
        # Ensure batch size is divisible by pac_size
        effective_batch_size = (batch_size // self.pac_size) * self.pac_size
        
        if effective_batch_size == 0:
            return torch.cat([data, cond_vec], dim=1) if cond_vec is not None else data
        
        # Reshape data for PacGAN
        data_pac = data[:effective_batch_size].view(-1, self.pac_size * data.size(1))
        
        if cond_vec is not None:
            cond_pac = cond_vec[:effective_batch_size].view(-1, self.pac_size * cond_vec.size(1))
            return torch.cat([data_pac, cond_pac], dim=1)
        else:
            return data_pac
    
    def _calculate_conditional_loss(self, fake_data, cond_vec, col_idx):
        """Calculate conditional loss for the generator"""
        if cond_vec is None or col_idx is None:
            return 0.0
        
        # Find the discrete column in the fake data
        data_ptr = 0
        for i, info in enumerate(self.data_transformer.output_info):
            if info['type'] == 'discrete' and i == col_idx:
                # Extract the generated discrete column
                n_categories = info['size']
                fake_discrete = fake_data[:, data_ptr:data_ptr + n_categories]
                
                # Extract the target from conditional vector
                cond_start = sum(info['size'] for info in self.data_transformer.output_info[:i] if info['type'] == 'discrete')
                target_discrete = cond_vec[:, cond_start:cond_start + n_categories]
                
                # Convert one-hot target to class indices
                target_indices = torch.argmax(target_discrete, dim=1)
                loss = nn.functional.cross_entropy(fake_discrete, target_indices)
                return loss
            
            if info['type'] == 'discrete':
                data_ptr += info['size']
            else:
                data_ptr += 1 + info['modes']
        
        return 0.0
    
    def sample(self, n_samples):
        """Generate synthetic samples using the trained model"""
        if self.generator is None:
            raise ValueError("Model must be trained before sampling")
        
        synthetic_samples = []
        
        with torch.no_grad():
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            for _ in range(n_batches):
                current_batch_size = min(self.batch_size, n_samples - len(synthetic_samples) * self.batch_size)
                
                # Sample conditional vector if we have discrete columns
                cond_vec, _, _, _ = self._sample_condvec(current_batch_size)
                
                # Generate noise
                noise = torch.randn(current_batch_size, 128).to(self.device)
                
                # Generate fake data
                fake_raw = self.generator(noise, cond_vec)
                fake_data = self._apply_activate(fake_raw)
                
                synthetic_samples.append(fake_data.cpu().numpy())
        
        # Concatenate all samples
        synthetic_data = np.concatenate(synthetic_samples)[:n_samples]
        
        # Transform back to original space
        synthetic_df = self.data_transformer.inverse_transform(synthetic_data)
        
        return synthetic_df
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves and privacy consumption"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training losses
        ax1.plot(self.training_losses['generator'], label='Generator', alpha=0.8)
        ax1.plot(self.training_losses['discriminator'], label='Discriminator', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Privacy consumption
        if self.privacy_history:
            ax2.plot(self.privacy_history, color='red', linewidth=2, label='Privacy Spent')
            ax2.axhline(y=self.epsilon, color='red', linestyle='--', 
                       label=f'Privacy Budget (ε={self.epsilon})')
            ax2.fill_between(range(len(self.privacy_history)), 
                           self.privacy_history, alpha=0.3, color='red')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Privacy Spent (ε)')
        ax2.set_title('Privacy Consumption (Opacus)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_privacy_spent(self):
        """Get the privacy budget spent"""
        return self.privacy_spent
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.generator is None:
            raise ValueError("No trained model to save")
        
        model_state = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'data_transformer': self.data_transformer,
            'training_losses': self.training_losses,
            'privacy_history': self.privacy_history,
            'privacy_spent': self.privacy_spent,
            'hyperparameters': {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'max_grad_norm': self.max_grad_norm,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'generator_lr': self.generator_lr,
                'discriminator_lr': self.discriminator_lr,
                'noise_multiplier': self.noise_multiplier,
                'pac_size': self.pac_size,
                'tau': self.tau
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model_state, filepath)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_state = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        hyperparams = model_state['hyperparameters']
        self.epsilon = hyperparams['epsilon']
        self.delta = hyperparams['delta']
        self.max_grad_norm = hyperparams['max_grad_norm']
        self.epochs = hyperparams['epochs']
        self.batch_size = hyperparams['batch_size']
        self.generator_lr = hyperparams['generator_lr']
        self.discriminator_lr = hyperparams['discriminator_lr']
        self.noise_multiplier = hyperparams.get('noise_multiplier')
        self.pac_size = hyperparams.get('pac_size', 10)
        self.tau = hyperparams.get('tau', 0.2)
        
        # Restore training history
        self.training_losses = model_state['training_losses']
        self.privacy_history = model_state['privacy_history']
        self.privacy_spent = model_state['privacy_spent']
        self.data_transformer = model_state['data_transformer']
        
        if self.verbose:
            print(f"Model loaded from {filepath}")
            print(f"Privacy spent: {self.privacy_spent:.3f}/{self.epsilon}")
    
    def __str__(self):
        return (f"OpacusDifferentiallyPrivateCTGAN("
               f"ε={self.epsilon}, δ={self.delta}, "
               f"epochs={self.epochs}, batch_size={self.batch_size}, "
               f"pac_size={self.pac_size}, tau={self.tau}, "
               f"privacy_spent={self.privacy_spent:.3f})") 