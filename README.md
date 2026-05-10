# Multilingual Hate Speech Detection in Nigerian Social Media

**A benchmarking study across Nigerian Pidgin, Hausa, Yoruba, and Igbo using the AfriHate Corpus**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cF3pkDbho45M8muN540H2VoA1aTLiy0g)
[![Dataset](https://img.shields.io/badge/Dataset-AfriHate-orange.svg)](https://huggingface.co/datasets/afrihate/afrihate)
[![Model](https://img.shields.io/badge/Model-AfriBERTa-green.svg)](https://huggingface.co/castorini/afriberta_large)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

This project is the implementation component of my final year dissertation at Miva Open University, Abuja (DTS 497). It conducts the first Nigeria-focused benchmarking analysis of automated hate speech detection across the four major Nigerian indigenous languages present in the AfriHate corpus (Muhammad et al., 2025).

Two classification models are evaluated under identical experimental conditions for each language:

- **TF-IDF + Logistic Regression**: a surface-level lexical baseline
- **Fine-tuned AfriBERTa**: a transformer pre-trained on 11 African languages

The study addresses a gap left by the AfriHate paper itself: no prior work had conducted a Nigeria-specific comparative analysis of these four language subsets, and no TF-IDF baseline had been reported for them.

---

## Languages Covered

| Language | Code | Subset Size | Hate Class | IAA (kappa) |
|---|---|---|---|---|
| Nigerian Pidgin | pcm | 10,599 | 11.1% | 0.65 |
| Hausa | hau | 6,644 | 5.2% | 0.75 |
| Igbo | ibo | 5,003 | 5.0% | 0.80 |
| Yoruba | yor | 4,879 | 3.1% | 0.68 |

---

## Key Results

| Language | TF-IDF LR (F1-Macro) | AfriBERTa (F1-Macro) |
|---|---|---|
| Hausa | 0.785 | **0.888** |
| Igbo | **0.956** | 0.949 |
| Nigerian Pidgin | **0.741** | 0.739 |
| Yoruba | **0.510** | 0.465 |
| Average | 0.748 | 0.760 |

**Main finding:** AfriBERTa's contextual pre-training provides meaningful gains only for Hausa (+0.103 F1-Macro). For the other three languages, the surface-level TF-IDF baseline performs comparably or better. Yoruba shows complete Hate-class detection failure across both models due to severe class imbalance (30 Hate test instances).

---

## Project Structure

```
nigeria-multilingual-hate-speech-detection/
│
├── DTS497_Project_Implementation.ipynb   # Main Colab notebook (17 cells)
├── README.md
│
└── outputs/                              # Generated after running the notebook
    ├── all_results.json                  # Full results for all language-model combinations
    ├── results_table.csv                 # Formatted results table
    ├── comparative_results.png           # F1-Macro and Accuracy bar charts
    ├── perclass_f1_heatmap.png           # Per-class F1 heatmap
    ├── class_distribution.png            # Class distribution by language
    ├── training_curves.png               # AfriBERTa val F1 and loss per epoch
    ├── cm_Hausa_TF-IDF_LR.png           # Confusion matrices (one per language)
    ├── cm_Igbo_TF-IDF_LR.png
    ├── cm_Nigerian_Pidgin_TF-IDF_LR.png
    ├── cm_Yoruba_TF-IDF_LR.png
    ├── error_analysis_hau.csv            # Misclassified instances per language
    ├── error_analysis_ibo.csv
    ├── error_analysis_pcm.csv
    └── error_analysis_yor.csv
```

---

## Getting Started

### Requirements

- Google account (for Colab)
- HuggingFace account with access to the AfriHate dataset
- T4 GPU runtime (free tier on Colab is sufficient)

### Step 1: Get dataset access

Visit [https://huggingface.co/datasets/afrihate/afrihate](https://huggingface.co/datasets/afrihate/afrihate), log in, and agree to the dataset terms. Then generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Step 2: Open the notebook

Click the Colab badge at the top of this README, or open it directly:

[https://colab.research.google.com/drive/1cF3pkDbho45M8muN540H2VoA1aTLiy0g](https://colab.research.google.com/drive/1cF3pkDbho45M8muN540H2VoA1aTLiy0g)

### Step 3: Set runtime

Go to Runtime > Change Runtime Type > Hardware Accelerator > T4 GPU.

### Step 4: Run cells in order

| Cell | Purpose | Time |
|---|---|---|
| 0 | Create outputs folder | Instant |
| 1 | Install dependencies, restart runtime | 2 min |
| 2 | Imports, config, HuggingFace login | Instant |
| 3 | Load AfriHate language subsets | 3-5 min |
| 4 | Preprocess all four languages | 3 min |
| 5-6 | Evaluation and visualisation helpers | Instant |
| 7 | TF-IDF + LR training (all 4 languages) | 10-15 min |
| 8-9 | AfriBERTa tokenisation and metrics helper | 5 min |
| 10 | AfriBERTa fine-tuning (all 4 languages) | 90-120 min |
| 11-16 | Results, charts, error analysis | 10 min |

### Step 5: Save outputs

After Cell 16 completes, run this to copy everything to Google Drive:

```python
import shutil, os
drive_path = '/content/drive/MyDrive/DTS497_Outputs'
os.makedirs(drive_path, exist_ok=True)
shutil.copytree('outputs', drive_path, dirs_exist_ok=True)
print("Saved to Google Drive.")
```

---

## Notable Implementation Details

**Why TF-IDF before AfriBERTa?**
The AfriHate paper (Muhammad et al., 2025) evaluated several large transformer and LLM-based models but did not include a TF-IDF baseline. This study fills that gap to quantify how much detection performance comes from surface-level lexical patterns versus contextual pre-trained representations.

**Class imbalance handling**
Both models use balanced class weights during training. For AfriBERTa, a custom `WeightedTrainer` subclass overrides `compute_loss()` to apply class-weighted cross-entropy, giving the Hate class proportionally higher loss weight.

**A dataset quirk worth knowing**
The AfriHate Nigerian subsets, as distributed on HuggingFace at the time of this study, contained only two effective label classes (Normal and Hate). The Abusive class was absent from all four language splits. This turned the study into an effective binary classification task. This is documented transparently in the thesis and reported here for reproducibility.

**Pidgin preprocessing**
Nigerian Pidgin has non-standardised orthography. A 30-item spelling variant dictionary normalises common variants (e.g. 'wia' to 'where', 'dem dem' to 'dem') before feature extraction. This reduces vocabulary fragmentation and improves TF-IDF coverage for Pidgin tweets.

---

## Dataset and Model References

**AfriHate Corpus**
Muhammad, S. H., Abdulmumin, I., Ayele, A. A., Adelani, D. I., et al. (2025). AfriHate: A multilingual collection of hate speech and abusive language datasets for African languages. In *Proceedings of NAACL 2025* (pp. 1854-1871). https://aclanthology.org/2025.naacl-long.92/

**AfriBERTa**
Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. In *Proceedings of the First Workshop on Multilingual Representation Learning* (pp. 116-126). https://aclanthology.org/2021.mrl-1.11/

---

## About This Project

**Author:** Solomon Ayuba 
**Institution:** Department of Data Science, School of Computing, Miva Open University, Abuja, Nigeria  
**Programme:** BSc (Hons) Data Science  
**Year:** 2026  
**Thesis title:** Multilingual Abusive Language Detection in Nigerian Social Media Using NLP: A Focused Benchmarking Study on Nigerian Pidgin, Hausa, Yoruba, and Igbo Using the AfriHate Corpus

---

## Acknowledgements

Thanks to Dr. Saminu Mohammad Aliyu and the AfriHate team for making their dataset publicly available, and to the Masakhane community for the African NLP infrastructure that made this work possible.

---

## License

This project is released under the MIT License. The AfriHate dataset is subject to its own terms at https://huggingface.co/datasets/afrihate/afrihate.
