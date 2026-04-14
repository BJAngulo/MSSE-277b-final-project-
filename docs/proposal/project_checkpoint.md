# EEG-Based Classification of Major Depressive Disorder (MDD)
### MSSE 277B Final Project — Checkpoint

---

## Team Members
Riya Tiloda, 

---

## Objective and Goal

Major Depressive Disorder (MDD) affects over 280 million people globally yet remains largely diagnosed through subjective clinical assessment. EEG offers a non-invasive, objective window into brain activity that may capture neurophysiological signatures of MDD.

**Primary goal:** Determine whether EEG-based MDD classifiers generalize across recording conditions — i.e., can a model trained on eyes-closed resting-state data correctly identify MDD patients in an eyes-open or P300 cognitive task recording?

**Secondary goal:** Use data augmentation (standard masking techniques vs. VAE-based) to assess whether synthetic EEG samples improve classification performance, and apply channel masking to identify which brain regions carry the most condition-robust signal.

---

## EDA Results

**Dataset:** nm000114 — 64 subjects (30 healthy controls, 34 MDD patients), each recorded under 3 conditions.

| Property | Value |
|---|---|
| Total recordings | 181 EDF files |
| Healthy subjects (sub-HS*) | 30 |
| MDD subjects (sub-MDDS*) | 34 |
| Conditions | eyesClosed (58), eyesOpen (62), P300 (61) |
| EEG channels (standardized) | 20 (10-20 system) |
| Channel mismatch | 112 files had 22 channels; 2 auxiliary channels dropped |
| Feature representation | 80-dim vector: 20 channels × 4 frequency bands |
| Frequency bands | delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) |

Band power was extracted using Welch's method (MNE) and normalized by total power to yield relative band power per channel. Patient-level GroupKFold splits are used throughout to prevent data leakage across recordings from the same subject.

---

## Project Outline

**Phase 1 — Baseline (complete)**
- Load and parse 181 EDF recordings; build metadata table
- Standardize to 20 common channels; extract 80-dim band-power features
- Establish within-condition cross-validated baseline (Logistic Regression, SVM) using GroupKFold

**Phase 2 — Cross-Condition Generalization**
- Train on one condition, test on each other condition (all 6 train/test pairs)
- Compare cross-condition accuracy to within-condition baseline
- Identify which condition produces the most transferable features

**Phase 3 — Augmentation Comparison**
- Standard augmentation: time masking, Gaussian noise, amplitude scaling
- VAE-based augmentation: train on training fold only, generate synthetic samples, evaluate classifier improvement
- Compare both approaches; report which (if either) meaningfully improves generalization

**Phase 4 — Interpretability via Channel Masking**
- Systematically mask each of the 20 channels and measure accuracy drop
- Identify the most predictive brain regions for MDD classification across conditions

---

## References

1. Kessler, R.C. et al. (2003). The epidemiology of major depressive disorder. *JAMA*, 289(23), 3095–3105.
2. Kingma, D.P. & Welling, M. (2013). Auto-encoding variational Bayes. *arXiv:1312.6114*.
3. Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
4. Mumtaz, W. et al. (2017). A review on EEG-based methods for detection and diagnosis of major depressive disorder. *Biomedical Signal Processing and Control*, 38, 198–213.
5. nm000114 dataset — OpenNeuro EEG MDD dataset (BIDS format).
6. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
