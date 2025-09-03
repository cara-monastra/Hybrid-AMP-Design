# Hybrid-AMP-Design
Hybrid AMP Design: Single-Driver Notebook

This repository contains a single Jupyter notebook, full single driver.ipynb, that implements three antimicrobial peptide (AMP) design strategies and evaluates them under a common set of filters and proxy metrics:

Motif-preserving interpolation

Property-conditioned cVAE (FiLM-conditioned encoder and decoder)

Hybrid approach that combines both

The notebook generates candidates, filters them to favor four-helix bundle feasibility, and reports side-by-side comparisons of predicted activity, predicted toxicity, novelty, and foldability proxies.

1. What the notebook produces

After a full run you will have:

Three candidate sets: Interpolation, cVAE, Hybrid

A consolidated results table with counts and per-arm means for:

activity, toxicity, novelty, helix_fraction, hydrophobic_moment, composite score

Figures:

Yield and pass-rate comparison

Activity and toxicity comparisons

Novelty and foldability distributions

cVAE training curves (reconstruction, KL, auxiliary loss)

Saved artifacts:

CSV and JSON files with per-sequence properties and scores

A “methods JSON” that records configuration for reproducibility

The results you saw previously were: Interpolation 2,000 raw (1,991 pass), cVAE 34 raw (32 pass), Hybrid 41 raw (39 pass), followed by the metric means reported in your Results section.

2. Software and hardware requirements

Python 3.11

Jupyter Notebook or JupyterLab

PyTorch (GPU optional; CPU will work but will be slower)

NumPy, pandas, matplotlib

If you have CUDA available, PyTorch will select it automatically; the notebook prints the chosen device at the top (“Device: cuda” or “Device: cpu”).

3. Data required

AMPS2.xlsx in the same directory as the notebook.
This file contains the curated AMP sequences used for training, validation, and property computation. The notebook expects a column named Sequence with peptide strings of canonical amino acids.

The notebook filters out sequences longer than 68 residues to avoid excessive padding. After filtering it uses 416 sequences for training and 178 for validation.

4. How to run

Create and activate an environment with the packages above.

Place AMPS2.xlsx next to full single driver.ipynb.

Open the notebook and run cells from top to bottom.

Confirm the setup cells print your device and that the dataset summary shows the expected split (Train n=416, Val n=178, MAX_LEN=68).

A full run trains the cVAE, builds the interpolation generator, runs all three arms, filters candidates, computes metrics, and renders plots.

5. What each section does

Setup and tokenization
Defines the 20-residue alphabet plus special tokens (PAD, BOS, EOS), builds stoi and itos, pads or truncates sequences to length 68.

Physicochemical properties
Computes net charge (pH 7.4), Kyte–Doolittle hydrophobicity, Chou–Fasman helix propensity, hydrophobic moment on an α-helix wheel, an amphipathicity index, and an interfacial transfer energy using a Wimley–White style proxy. Features are normalized and concatenated as conditioning vectors.

cVAE
Bidirectional LSTM encoder, 64-dimensional latent, FiLM modulation in encoder and decoder, autoregressive LSTM decoder, trained for 40 epochs with Adam at 1e-3. Loss combines reconstruction cross-entropy, KL with cyclical beta and free-bits, plus a light auxiliary regression from latent to properties.

Interpolation generator
Interpolates between property vectors of training pairs, assembles candidates while preserving motifs and enforcing property targets, then filters on length, charge, helix fraction, and hydrophobic moment.

Hybrid arm
Uses interpolation candidates as pseudo-training augmentation and decodes additional sequences from the cVAE under interpolated conditions. Applies the same filters as interpolation.

Evaluation
Computes novelty via k-mer divergence from the training set, predicted activity via a property-based heuristic, predicted toxicity via conservative thresholds on hydrophobicity, charge, length, and tryptophan content, and foldability proxies via helix fraction and hydrophobic moment. Consolidates results into a single table and renders plots.

Exports
Saves per-sequence annotations (CSV and JSON) and a methods JSON with configuration for reproducibility.

6. Reproducing the reported numbers

Use the default hyperparameters in the notebook:

MAX_LEN 68, latent 64, epochs 40, lr 1e-3

Interpolation target settings as provided in the notebook

Interpolation n_generate = 2000

Ensure AMPS2.xlsx has the expected sequences and a Sequence column.

Run all cells without changing the filters:

length within ±3 of target

charge near +6

helix fraction above 0.6

hydrophobic moment around 1.5

You should recover the same order of magnitude counts and similar mean metric values. Exact decimals can drift slightly due to random seeds and GPU nondeterminism.

7. How to tweak experiments

Change throughput
In the interpolation section, set n_generate to any integer. For example, n_generate = 5000 will produce five thousand candidates before filtering.

Adjust property targets
Modify the target_props dictionary to steer length, charge, hydrophobicity, helix fraction, hydrophobic moment, amphipathicity, or interfacial energy. This affects both interpolation and the conditioning given to the cVAE.

Tighten or relax filters
The pass rate is sensitive to thresholds. Raising the helix fraction cutoff or lowering the acceptable hydrophobic moment band will reduce yield and bias toward strongly amphipathic helices.

Encourage more cVAE samples
Increase the number of latent samples per condition or the number of conditions evaluated. You can also adjust the cyclical KL schedule, free-bits threshold, or sampling temperature to trade novelty against pass rate.

8. Interpreting the metrics

Predicted activity is a heuristic built from net charge, hydrophobicity, and amphipathicity. Use it to compare arms within this project, not as a substitute for MIC assays.

Predicted toxicity reflects conservative rules drawn from AMP toxicology literature. Sequences that violate several thresholds should be viewed as risky for hemolysis.

Novelty is k-mer divergence. Pair it with a global diversity measure if you plan clustering or down-selection.

Foldability proxies (helix fraction, hydrophobic moment) help assess helical amphipathicity relevant to four-helix bundle designs.

9. Troubleshooting

Notebook cannot find AMPS2.xlsx
Place the file in the same directory as the notebook and confirm the sheet has a Sequence column.

CUDA not used
If the device prints “cpu,” verify that your PyTorch install matches your CUDA version. The notebook still runs on CPU.

Zero or very low cVAE yield
Relax filters slightly, reduce KL pressure early in training, or extend training epochs. Ensure conditioning vectors are normalized consistently between training and generation.

All-Ala or collapsed sequences
Lower label smoothing if used, increase free-bits slightly, and verify that BOS/EOS handling and PAD masking are correct in the decoder loss.

10. Reuse and citation

If you use results or code derived from this notebook in a manuscript or dissertation, please cite your thesis and acknowledge the use of a property-conditioned VAE with FiLM and a motif-preserving interpolation pipeline, along with the AMP property scales used for conditioning (Kyte–Doolittle, Chou–Fasman, Wimley–White).
