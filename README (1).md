# EIS — Embedding Instability Score for Backdoor Detection in NLP

Detecting backdoor attacks in BERT-based NLP models using geometry of intermediate transformer layer embeddings. No trigger knowledge required. No training data access required.

---

## The Problem

Backdoor attacks on NLP models are invisible to standard monitoring. An adversary poisons 1–5% of training data with a trigger token and achieves 100% attack success rate while accuracy drops by less than 0.5%. You cannot catch this by watching accuracy.

## The Solution

EIS measures Jensen-Shannon divergence between pairwise Euclidean distance distributions of CLS token embeddings at each transformer layer. A backdoor creates a geometric attractor at mid-semantic layers (L6–L8 in BERT) that is detectable without knowing the trigger.

```
EIS(L) = JSD( D_clean(L) || D_poisoned(L) )
```

---

## Results

### Attack Stealth (why accuracy monitoring fails)

| Dataset | Poison Rate | Clean Acc | Poisoned Acc | Drop | ASR |
|---|---|---|---|---|---|
| IMDB | 1% | 0.8907 | 0.8896 | 0.0011 | 1.00 |
| IMDB | 5% | 0.9005 | 0.8983 | 0.0022 | 1.00 |
| SST-2 | 2% | 0.9523 | 0.9485 | 0.0039 | 1.00 |
| DistilBERT IMDB | 1% | 0.8807 | 0.8799 | 0.0008 | 1.00 |
| RoBERTa IMDB | 1% | 0.8820 | 0.8720 | 0.0100 | 0.48 |

### EIS Layer Profile — BERT, IMDB, 1% Poison

| Layer | EIS Score | Note |
|---|---|---|
| L4 | 0.001081 | Early — syntactic, minimal distortion |
| L6 | 0.029666 | Rising |
| L8 | 0.262514 | **Peak — geometric attractor** |
| L12 | 0.027653 | Low despite highest vector displacement (L12 paradox) |

> **L12 Paradox:** Mean absolute vector displacement is highest at L12 (0.436) but EIS is lowest. Both clean and poisoned models converge to the same output geometry. The backdoor hides at the final layer. Detectors using final-layer representations miss it entirely.

### Baseline Comparison — IMDB, 5% Poison, BERT-base-uncased

| Method | Detection Rate | FPR | AUC-ROC |
|---|---|---|---|
| STRIP | 0.8700 | 0.5633 | 0.7294 |
| Spectral Signatures | 1.0000 | 0.5000 | 0.5145 |
| ONION | 0.8200 | 0.5567 | 0.6655 |
| **EIS (Ours, L6)** | **1.0000** | **0.0000** | **1.0000** |

**Why baselines fail:**
- STRIP: OOV trigger `cfhjq` → tokenized as `[UNK]` → minimal entropy separation
- Spectral Signatures: operates at L12 — the exact layer where the signal is suppressed
- ONION: GPT-2 perplexity partially catches OOV tokens but 56% FPR makes it unreliable

---

## Architectures Tested

| Model | Layers | Peak EIS Layer | Behaviour |
|---|---|---|---|
| BERT-base-uncased | 12 | L6–L8 | Strong mid-layer signal |
| DistilBERT-base | 6 | L4–L5 | Peak at proportional depth |
| RoBERTa-base | 12 | None | Near-zero — backdoor resistant |

---

## Datasets

All loaded via HuggingFace `datasets` library:
- IMDB (50k reviews)
- SST-2 (Stanford Sentiment Treebank)
- Yelp Polarity

---

## Requirements

```
transformers>=4.0
datasets
torch
scikit-learn
scipy
matplotlib
tqdm
```

---

## Usage

1. Upload `EIS_IMDB_Baselines_Finalrr.ipynb` to Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells sequentially
4. IMDB CSV must be uploaded to `/content/imdb.csv`

---

## Authors

Raunit Jain, Khilan Kanadia, Jinay Jain, Raaj Shah, Dr. Kranti Ghar
Dwarkadas J. Sanghvi College of Engineering, Mumbai, India

## Status

Paper in preparation for submission — IAES International Journal of Artificial Intelligence (IJ-AI), Q2 Scopus.
