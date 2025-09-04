# Hybrid-AMP-Design

A comprehensive Jupyter notebook implementation for antimicrobial peptide (AMP) design using three distinct strategies with unified evaluation metrics.

## Overview

This repository contains `full single driver.ipynb`, which implements and compares three AMP design approaches:

- **Motif-preserving interpolation**
- **Property-conditioned cVAE** (FiLM-conditioned encoder and decoder)
- **Hybrid approach** combining both methods

The notebook generates peptide candidates, applies filters optimized for four-helix bundle feasibility, and provides comprehensive side-by-side comparisons of predicted activity, toxicity, novelty, and foldability metrics.

##  What You'll Get

After running the complete notebook:

### Generated Datasets
- **Three candidate sets**: Interpolation, cVAE, and Hybrid approaches
- **Consolidated results table** with per-method statistics

### Key Metrics
- Activity and toxicity predictions
- Novelty scores and foldability proxies
- Helix fraction and hydrophobic moment analysis
- Composite scoring system

### Visualizations
- Yield and pass-rate comparisons
- Activity vs. toxicity scatter plots
- Novelty and foldability distributions
- cVAE training curves (reconstruction, KL, auxiliary loss)

### Saved Outputs
- CSV and JSON files with per-sequence properties
- Methods JSON for full reproducibility

##  Requirements

### Software
- **Python 3.11**
- **Jupyter Notebook** or JupyterLab
- **PyTorch** (GPU optional, CPU supported)
- **Standard packages**: NumPy, pandas, matplotlib

> **Note**: If CUDA is available, PyTorch will automatically use GPU acceleration. The notebook displays the selected device at startup.

### Data
- `AMPS2.xlsx` must be placed in the same directory as the notebook
- File should contain a `Sequence` column with canonical amino acid peptide strings
- Sequences longer than 68 residues are automatically filtered out
- Expected dataset split: 416 training sequences, 178 validation sequences

##  Quick Start

1. **Setup environment** with required packages
2. **Place `AMPS2.xlsx`** next to `full single driver.ipynb`
3. **Open the notebook** in Jupyter
4. **Run cells sequentially** from top to bottom
5. **Verify setup** by checking device output and dataset summary

Expected output:
```
Device: cuda  # or cpu
Train n=416, Val n=178, MAX_LEN=68
```

## 📖 Notebook Structure

### Core Components

**Setup & Tokenization**
- 20-residue alphabet plus special tokens (PAD, BOS, EOS)
- Sequence padding/truncation to length 68

**Physicochemical Properties**
- Net charge calculation (pH 7.4)
- Kyte–Doolittle hydrophobicity
- Chou–Fasman helix propensity
- Hydrophobic moment and amphipathicity indices
- Wimley–White interfacial transfer energy proxy

**cVAE Architecture**
- Bidirectional LSTM encoder with 64D latent space
- FiLM modulation in encoder/decoder
- Autoregressive LSTM decoder
- Training: 40 epochs, Adam optimizer (lr=1e-3)
- Loss: reconstruction + KL (cyclical beta, free-bits) + auxiliary regression

**Generation Methods**
- **Interpolation**: Property-guided interpolation with motif preservation
- **Hybrid**: cVAE decoding with interpolated conditioning

**Evaluation Pipeline**
- Novelty via k-mer divergence
- Activity/toxicity heuristics
- Foldability proxies
- Comprehensive filtering and scoring

##  Reproducing Results

Use these default parameters:
- `MAX_LEN = 68`
- `latent_dim = 64` 
- `epochs = 40`
- `learning_rate = 1e-3`
- `n_generate = 2000` (interpolation)

### Filtering Criteria
- Length: target ±3 residues
- Charge: ~+6
- Helix fraction: >0.6
- Hydrophobic moment: ~1.5

**Expected Results**: Interpolation ~2,000 candidates (1,991 pass), cVAE ~34 candidates (32 pass), Hybrid ~41 candidates (39 pass)

## Customization Options

### Adjust Generation Volume
```python
n_generate = 5000  # Increase candidate pool
```

### Modify Property Targets
```python
target_props = {
    'length': 25,
    'charge': +8,
    'hydrophobicity': 0.3,
    # ... customize other properties
}
```

### Fine-tune Filters
- **Stricter**: Increase helix fraction threshold
- **Looser**: Widen hydrophobic moment acceptance range
- **Novel**: Adjust sampling temperature and KL schedule

## Metric Interpretation

| Metric | Purpose | Notes |
|--------|---------|-------|
| **Predicted Activity** | Heuristic scoring | For comparative analysis, not MIC replacement |
| **Predicted Toxicity** | Hemolysis risk assessment | Based on literature thresholds |
| **Novelty** | k-mer divergence from training | Combine with diversity measures |
| **Foldability** | Helix fraction + hydrophobic moment | Relevant for four-helix bundles |

##  Troubleshooting

### Common Issues

**File Not Found**: Ensure `AMPS2.xlsx` is in the notebook directory with `Sequence` column

**CPU-Only Execution**: Verify PyTorch-CUDA version compatibility (notebook runs on CPU regardless)

**Low cVAE Yield**: 
- Relax filtering thresholds
- Reduce early KL pressure  
- Extend training epochs
- Check conditioning vector normalization

**Degenerate Sequences**: 
- Increase free-bits threshold
- Verify BOS/EOS handling
- Check PAD masking in decoder loss



3. **Reference scales**: Kyte–Doolittle, Chou–Fasman, Wimley–White property scales


