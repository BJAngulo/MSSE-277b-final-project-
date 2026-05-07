# EEG-MDD Classification 

This project compares classical machine learning and deep learning approaches for EEG-based classification of Major Depressive Disorder (MDD). 

The workflow evaluates:
 
- Welch band-power feature models
- Raw EEG baselines (PCA + SVM)
- Deep learning models (CNN, LSTM, CNN-LSTM)
- Temporal sensitivity via ablation 

---

## Project Structure
```text
data/
├── raw/                  # Original EDF files
└── segmented_data/       # Cached segmented EEG arrays

results/
├── tables/                 # Model comparison results
└── figures/                # Plots and visualizations

src/
├── config.py               # Paths and experiment settings
├── metadata.py             # Build metadata from EDF files
├── features.py             # Welch feature extraction + preprocessing
├── segmentation.py         # Segmentation + caching logic
├── data_loading.py         # Load cached datasets/results
├── evaluation.py           # Cross-validation + metrics
├── models.py               # Classical ML models
├── deep_learning.py        # Load cached deep model results
└── run_analysis.py         # Main analysis pipeline 

scripts/
└── train_deep_models.py    # Deep model retraining


Makefile
requirements.txt
README.md 
```

## ⚙️ Environment Setup

```bash
make env
conda activate msse_277b_final_project
```

## Running the analysis 
```bash
make run 
```
or 
```bash
python -m src.run_analysis
```

This will:
- Load cached segmented EEG datasets
- Load saved model results
- Rebuild the unified model comparison table
- Save results to: 
 ```bash
results/tables/unified_model_comparison.csv
```

## Deep Learning Models
Deep learning models (CNN, LSTM, CNN-LSTM) are not retrained by default to avoid long runtimes. 

To retrain them:
```bash 
make train-deep
```
## Key Results 
- Best model: Multi-Scale CNN (ROC-AUC~0.93)
- Strong baseline: Welch + SVM(ROC-AUC~0.91)
- Weak baseline: Raw + PCA + SVM (ROC-AUC~0.81)

### Key findings
- EEG signals contain nonlinear structure (UMAP > PCA)
- CNNs effectively capture localized patterns
- Increasing model complexity (deep CNN, LSTM) does not improve performance
- Temporal information is distributed rather than localized

## Analysis Components

### Feature-based models
- Welch PSD -> band-power features
- Logistic Regression, SVM

### Raw signal models
- PCA + SVM baseline
- CNN / Deep CNN / MultiScale CNN
- LSTM / CNN-LSTM

### Temporal analysis
- Window size sensitivity
- Temporal ablation 


### Notes 
- Segmeneted datasets are cached in "data/segmented_data/"
- Delete cached files if preprocessing changes:
```bash
make clean-cache

```
## Summary 
 This project shows that: 
    EEG classification is driven by localized nonlinear patterns, best captured by CNNs, while increasing model complexity or temporal depth does not improve performance. 
