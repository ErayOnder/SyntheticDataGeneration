# Synthetic Data Generation Evaluation

This project implements and evaluates differentially private synthetic data generation methods, specifically comparing PrivBayes and DP-CTGAN synthesizers across different privacy budgets.

## Project Overview

The project evaluates synthetic data quality using three main metrics:
- **Statistical Quality**: Marginal distributions, correlation preservation, and PMSE
- **ML Utility**: Train-on-Synthetic-Test-on-Real (TSTR) evaluation
- **Privacy Protection**: Exact match counting and Distance to Closest Record (DCR)

## Datasets

- **Adult**: UCI Adult dataset (income prediction)
- **Covertype**: UCI Covertype dataset (forest cover type classification)

## Synthesizers

- **PrivBayes**: Differentially private Bayesian network
- **DP-CTGAN**: Differentially private Conditional Tabular GAN

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ErayOnder/SyntheticDataGeneration.git
   cd SyntheticDataGeneration
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets** (automatically handled on first run):
   - Adult dataset will be downloaded to `data/`
   - Covertype dataset will be downloaded to `data/`

## Usage

### Quick Start

Run the evaluation pipeline with default settings:
```bash
python src/test_evaluation_pipeline.py --synthesizer privbayes --dataset adult
```

### Command Line Options

```bash
python src/test_evaluation_pipeline.py [OPTIONS]

Options:
  --synthesizer {privbayes,dpctgan}  Type of synthesizer (default: privbayes)
  --dataset {adult,covertype}        Dataset to use (default: adult)
  --train-size INT                   Number of training samples (default: 1000)
  --test-size INT                    Number of test samples (default: 500)
  --epsilon FLOAT                    Privacy budget(s), comma-separated (default: 1.0)
  --n-trials INT                     Number of trials to run (default: 3)
```

### Examples

**Evaluate PrivBayes on Adult dataset with multiple epsilon values**:
```bash
python src/test_evaluation_pipeline.py --synthesizer privbayes --dataset adult --epsilon 0.5,1.0,5.0
```

**Evaluate DP-CTGAN on Covertype dataset**:
```bash
python src/test_evaluation_pipeline.py --synthesizer dpctgan --dataset covertype --epsilon 1.0
```

**Quick test with smaller sample sizes**:
```bash
python src/test_evaluation_pipeline.py --synthesizer privbayes --dataset adult --train-size 500 --test-size 200 --n-trials 1
```

## Output

Results are saved in the `results/` directory with the following structure:
```
results/
├── {dataset}_{synthesizer}_eps_{epsilon}/
│   ├── aggregated_ml_metrics.json
│   ├── aggregated_privacy_metrics.json
│   ├── aggregated_statistical_metrics.json
│   ├── baseline_tstr_results.csv
│   ├── tstr_results.csv
│   ├── marginal_distributions.png
│   ├── correlation_difference.png
│   └── trial-specific metrics files...
```

## Project Structure

```
SyntheticDataGeneration/
├── data/                    # Dataset files
├── src/                     # Source code
│   ├── data_preprocessor.py # Data preprocessing
│   ├── synthesizers/        # Synthetic data generators
│   ├── evaluation/          # Evaluation metrics
│   └── test_evaluation_pipeline.py
├── results/                 # Evaluation results
├── notebooks/               # Jupyter notebooks
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.7+
- See `requirements.txt` for specific package versions 