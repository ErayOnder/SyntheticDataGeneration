import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Opacus imports for differential privacy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants.utils import get_noise_multiplier

# CTGAN imports
from ctgan import CTGAN
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer


class OpacusDifferentiallyPrivateCTGAN:
    """
    CTGAN implementation with Opacus integration for proper DP-SGD training
    
    This follows the model-first approach with formal differential privacy guarantees
    using Facebook's Opacus library for DP-SGD.
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, max_grad_norm=1.0, 
                 epochs=100, batch_size=500, generator_lr=2e-4, discriminator_lr=2e-4,
                 generator_decay=1e-6, discriminator_decay=1e-6, discriminator_steps=1,
                 verbose=True):
        """
        Initialize DP-CTGAN with Opacus integration
        
        Args:
            epsilon (float): Privacy budget parameter (smaller = more private)
            delta (float): Relaxation parameter for (ε,δ)-DP
            max_grad_norm (float): Maximum gradient norm for clipping
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            generator_lr (float): Generator learning rate
            discriminator_lr (float): Discriminator learning rate
            generator_decay (float): Generator weight decay
            discriminator_decay (float): Discriminator weight decay
            discriminator_steps (int): Discriminator steps per generator step
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
    
    def _create_generator(self, data_dim, cond_dim, latent_dim=128):
        """Create CTGAN generator network"""
        class Generator(nn.Module):
            def __init__(self, latent_dim, data_dim, cond_dim, hidden_dims=[256, 256]):
                super().__init__()
                input_dim = latent_dim + cond_dim
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU()
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, data_dim))
                self.model = nn.Sequential(*layers)
            
            def forward(self, noise, cond=None):
                if cond is not None:
                    input_tensor = torch.cat([noise, cond], dim=1)
                else:
                    input_tensor = noise
                return self.model(input_tensor)
        
        return Generator(latent_dim, data_dim, cond_dim).to(self.device)
    
    def _create_discriminator(self, data_dim, cond_dim):
        """Create CTGAN discriminator network"""
        class Discriminator(nn.Module):
            def __init__(self, data_dim, cond_dim, hidden_dims=[256, 256]):
                super().__init__()
                input_dim = data_dim + cond_dim
                
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
                self.model = nn.Sequential(*layers)
            
            def forward(self, data, cond=None):
                if cond is not None:
                    input_tensor = torch.cat([data, cond], dim=1)
                else:
                    input_tensor = data
                return self.model(input_tensor)
        
        return Discriminator(data_dim, cond_dim).to(self.device)
    
    def _prepare_data(self, data):
        """Prepare data using CTGAN's data transformer"""
        # Initialize data transformer
        self.data_transformer = DataTransformer()
        
        # Identify discrete columns
        discrete_columns = []
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].nunique() < 10:
                discrete_columns.append(col)
        
        if self.verbose:
            print(f"Discrete columns: {discrete_columns}")
        
        # Transform data
        transformed_data = self.data_transformer.fit_transform(data, discrete_columns)
        
        # Create data sampler
        self.data_sampler = DataSampler(
            data=transformed_data,
            output_info=self.data_transformer.output_info_list,
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
    
    def fit(self, data):
        """
        Train the DP-CTGAN using Opacus for DP-SGD
        """
        print("=" * 60)
        print("TRAINING DP-CTGAN WITH OPACUS")
        print("=" * 60)
        print(f"Privacy Parameters: ε={self.epsilon}, δ={self.delta}")
        print(f"Max Gradient Norm: {self.max_grad_norm}")
        print(f"Training Configuration: {self.epochs} epochs, batch size {self.batch_size}")
        print("=" * 60)
        
        # Prepare data
        transformed_data = self._prepare_data(data)
        dataset_size = len(data)
        
        # Compute privacy parameters
        self._compute_privacy_parameters(dataset_size)
        
        # Get data dimensions
        data_dim = sum([info.dim for info in self.data_transformer.output_info_list])
        cond_dim = self.data_sampler.dim_cond_vec()
        
        if self.verbose:
            print(f"Data dimension: {data_dim}")
            print(f"Conditional dimension: {cond_dim}")
        
        # Create models
        self.generator = self._create_generator(data_dim, cond_dim)
        self.discriminator = self._create_discriminator(data_dim, cond_dim)
        
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
        
        # Setup Opacus Privacy Engine for discriminator only
        # (Generator doesn't need DP since it doesn't see real data directly)
        self.privacy_engine = PrivacyEngine()
        
        self.discriminator, optimizer_d, train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.discriminator,
            optimizer=optimizer_d,
            data_loader=self._create_dataloader(transformed_data),
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )
        
        if self.verbose:
            print(f"Privacy engine initialized with ε={self.epsilon}, δ={self.delta}")
        
        # Training loop
        self._train_with_opacus(optimizer_g, optimizer_d, train_loader)
        
        print("Training completed!")
        print("=" * 60)
    
    def _create_dataloader(self, transformed_data):
        """Create PyTorch DataLoader for training"""
        # Convert to tensor
        data_tensor = torch.FloatTensor(transformed_data).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        return dataloader
    
    def _train_with_opacus(self, optimizer_g, optimizer_d, train_loader):
        """Training loop with Opacus privacy engine"""
        
        for epoch in tqdm(range(self.epochs), desc="Training DP-CTGAN"):
            g_losses = []
            d_losses = []
            
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=self.batch_size,
                optimizer=optimizer_d
            ) as memory_safe_data_loader:
                
                for batch_idx, (real_data,) in enumerate(memory_safe_data_loader):
                    batch_size = real_data.size(0)
                    
                    # Train Discriminator with DP
                    for _ in range(self.discriminator_steps):
                        optimizer_d.zero_grad()
                        
                        # Real data
                        # Sample conditional vector
                        if self.data_sampler.dim_cond_vec() > 0:
                            cond_vec = self.data_sampler.sample_condvec(batch_size)
                            if cond_vec is not None:
                                c1, m1, col, opt = cond_vec
                                c1 = torch.FloatTensor(c1).to(self.device)
                                real_data_cond = torch.cat([real_data, c1], dim=1)
                            else:
                                real_data_cond = real_data
                                c1 = None
                        else:
                            real_data_cond = real_data
                            c1 = None
                        
                        # Generate fake data
                        noise = torch.randn(batch_size, 128).to(self.device)
                        if c1 is not None:
                            fake_data = self.generator(noise, c1)
                            fake_data_cond = torch.cat([fake_data, c1], dim=1)
                        else:
                            fake_data = self.generator(noise)
                            fake_data_cond = fake_data
                        
                        # Discriminator predictions
                        real_pred = self.discriminator(real_data_cond)
                        fake_pred = self.discriminator(fake_data_cond.detach())
                        
                        # Wasserstein loss
                        d_loss = torch.mean(fake_pred) - torch.mean(real_pred)
                        
                        d_loss.backward()
                        optimizer_d.step()
                        
                        d_losses.append(d_loss.item())
                    
                    # Train Generator (no DP needed)
                    optimizer_g.zero_grad()
                    
                    noise = torch.randn(batch_size, 128).to(self.device)
                    if c1 is not None:
                        fake_data = self.generator(noise, c1)
                        fake_data_cond = torch.cat([fake_data, c1], dim=1)
                    else:
                        fake_data = self.generator(noise)
                        fake_data_cond = fake_data
                    
                    fake_pred = self.discriminator(fake_data_cond)
                    g_loss = -torch.mean(fake_pred)
                    
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
    
    def sample(self, n_samples):
        """Generate synthetic samples"""
        if self.generator is None:
            raise ValueError("Model must be trained before sampling")
        
        synthetic_samples = []
        
        with torch.no_grad():
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            for _ in range(n_batches):
                current_batch_size = min(self.batch_size, n_samples - len(synthetic_samples))
                
                # Generate noise
                noise = torch.randn(current_batch_size, 128).to(self.device)
                
                # Sample conditional vector if needed
                if self.data_sampler.dim_cond_vec() > 0:
                    cond_vec = self.data_sampler.sample_condvec(current_batch_size)
                    if cond_vec is not None:
                        c1, m1, col, opt = cond_vec
                        c1 = torch.FloatTensor(c1).to(self.device)
                        fake_data = self.generator(noise, c1)
                    else:
                        fake_data = self.generator(noise)
                else:
                    fake_data = self.generator(noise)
                
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
                'noise_multiplier': self.noise_multiplier
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
               f"privacy_spent={self.privacy_spent:.3f})") 