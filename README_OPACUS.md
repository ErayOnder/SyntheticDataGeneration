# Week 2: Opacus Differentially Private CTGAN Implementation

## 🎯 Project Overview

This implementation demonstrates the **model-first approach** for differentially private synthetic data generation using **CTGAN with Opacus integration** for formal DP-SGD training.

### Goals Achieved:
✅ **CTGAN-based synthetic data generator with differential privacy**  
✅ **Opacus integration for DP-SGD training**  
✅ **Model-first approach with formal privacy guarantees**

## 📁 Project Structure

```
├── dp_ctgan_opacus.py           # Main Opacus DP-CTGAN implementation
├── week2_opacus_pipeline.py     # Complete pipeline script
├── test_opacus_pipeline.py      # Comprehensive test suite
├── quick_test.py               # Quick verification script
├── data_preprocessor.py        # Adult dataset preprocessing
├── evaluation.py               # Synthetic data evaluation suite
├── requirements.txt            # Dependencies including Opacus
└── README_OPACUS.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Verification
```bash
python quick_test.py
```

### 3. Run Pipeline (Quick Mode)
```bash
python week2_opacus_pipeline.py --quick_mode
```

### 4. Full Training
```bash
python week2_opacus_pipeline.py --epsilon 1.0 --epochs 100 --n_samples 5000
```

## 🔧 Opacus Integration Features

### Core Opacus Components Used:
- **PrivacyEngine**: For formal DP-SGD guarantees
- **make_private_with_epsilon()**: Automatic privacy parameter setup
- **BatchMemoryManager**: Memory-efficient training for large models
- **get_epsilon()**: Real-time privacy accounting
- **get_noise_multiplier()**: Automatic noise calibration

### Implementation Highlights:
```python
# Privacy Engine Setup
self.privacy_engine = PrivacyEngine()
self.discriminator, optimizer_d, train_loader = self.privacy_engine.make_private_with_epsilon(
    module=self.discriminator,
    optimizer=optimizer_d,
    data_loader=train_loader,
    epochs=self.epochs,
    target_epsilon=self.epsilon,
    target_delta=self.delta,
    max_grad_norm=self.max_grad_norm,
)

# Training with Memory Management
with BatchMemoryManager(
    data_loader=train_loader,
    max_physical_batch_size=self.batch_size,
    optimizer=optimizer_d
) as memory_safe_data_loader:
    # DP-SGD training loop
```

## 📊 Usage Examples

### Basic Usage
```python
from dp_ctgan_opacus import OpacusDifferentiallyPrivateCTGAN

# Initialize with privacy parameters
dp_ctgan = OpacusDifferentiallyPrivateCTGAN(
    epsilon=1.0,
    delta=1e-5,
    max_grad_norm=1.0,
    epochs=100,
    batch_size=500
)

# Train on your data
dp_ctgan.fit(training_data)

# Generate synthetic data
synthetic_data = dp_ctgan.sample(n_samples=1000)

# Check privacy spent
privacy_used = dp_ctgan.get_privacy_spent()
```

### Command Line Options
```bash
python week2_opacus_pipeline.py [OPTIONS]

Options:
  --epsilon FLOAT        Privacy budget ε (default: 1.0)
  --delta FLOAT         Delta parameter δ (default: 1e-5)
  --max_grad_norm FLOAT Maximum gradient norm (default: 1.0)
  --epochs INT          Training epochs (default: 50)
  --batch_size INT      Batch size (default: 500)
  --n_samples INT       Synthetic samples to generate (default: 5000)
  --quick_mode          Quick mode for testing
```

## 🔒 Privacy Parameters

### Privacy Budget (ε)
- **ε ≤ 1.0**: High privacy protection
- **1.0 < ε ≤ 3.0**: Moderate privacy protection
- **ε > 3.0**: Lower privacy protection, higher utility

### Delta (δ)
- Typically set to 1/n where n is dataset size
- Default: 1e-5 (suitable for most datasets)

### Gradient Clipping
- Controls sensitivity of gradients
- Higher values: More information retained, less noise
- Lower values: More restrictive, more noise required

## 📈 Model Architecture

### CTGAN Components:
- **Generator**: Learns to generate realistic synthetic data
- **Discriminator**: Distinguishes real from synthetic data (DP-SGD applied here)
- **DataTransformer**: Handles mixed-type tabular data
- **DataSampler**: Manages conditional sampling

### Privacy Application:
- **Discriminator**: DP-SGD via Opacus (sees real data)
- **Generator**: No DP needed (doesn't see real data directly)
- **Privacy Accounting**: Tracked per-epoch via Opacus

## 📊 Evaluation Metrics

The pipeline includes comprehensive evaluation:

### Statistical Similarity
- Kolmogorov-Smirnov tests for continuous variables
- Jensen-Shannon divergence for categorical variables
- Overall similarity scores

### ML Efficacy
- Train on Synthetic, Test on Real (TSTR) paradigm
- Multiple ML models: Random Forest, Logistic Regression, SVM
- Utility preservation ratios

### Privacy Assessment
- Basic membership inference analysis
- Distance-based privacy metrics

## 🎯 Expected Results

### Quality Scores (0.0 - 1.0):
- **Statistical Similarity**: How well distributions match
- **ML Efficacy**: How well models trained on synthetic perform on real
- **Privacy Score**: Basic privacy preservation assessment
- **Overall Quality**: Weighted average of all metrics

### Privacy-Utility Trade-off:
- **High ε (low privacy)**: Better utility, lower privacy
- **Low ε (high privacy)**: Lower utility, better privacy
- **Optimal range**: ε ∈ [0.5, 2.0] for most applications

## 🔄 Troubleshooting

### Common Issues:
1. **Memory errors**: Reduce batch_size or use smaller datasets
2. **Version compatibility**: Ensure compatible Opacus/PyTorch versions
3. **Privacy budget exhausted**: Increase ε or reduce epochs

### Performance Tips:
- Use GPU for faster training if available
- Start with quick_mode for parameter tuning
- Monitor privacy consumption to avoid budget exhaustion

## 📚 References

- **Opacus**: Facebook's library for training PyTorch models with differential privacy
- **CTGAN**: Conditional Tabular GAN for synthetic data generation
- **DP-SGD**: Differentially Private Stochastic Gradient Descent

## 🏆 Verification

Run the quick test to verify everything works:
```bash
python quick_test.py
```

Expected output:
```
✅ All Opacus imports successful
✅ Opacus DP-CTGAN initialized successfully  
✅ Privacy parameters computed
🎯 GOALS VERIFICATION:
✅ CTGAN-based synthetic data generator with differential privacy: IMPLEMENTED
✅ Opacus integration for DP-SGD training: IMPLEMENTED
✅ Model-first approach with formal privacy guarantees: IMPLEMENTED
``` 