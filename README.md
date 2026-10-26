# Multimodal Skin Cancer Detection: Fusing Vision Transformers and XGBoost

**Author:** Kenji Wagner (team project for CS640: Artificial Intelligence, Boston University, Fall 2024)

**Dataset:** [ISIC 2024 Skin Cancer Detection with 3D-TBP (SLICE-3D)](https://www.kaggle.com/competitions/isic-2024-challenge). It contains 15×15 mm lesion crops from 3D total-body photography, each paired with clinical and lesion metadata (age, sex, body location, size, shape, color).

## Executive Summary

**Objective:** Classify skin lesions as malignant or benign by combining what a lesion *looks like* (the image) with what is *known about it* (the tabular metadata). The goal was a model that beats either source on its own, evaluated without the data leakage that often inflates medical-imaging results.

**The core challenge: extreme class imbalance.** Only **0.098%** of the ~320,000 training lesions are malignant, so a model that always says "benign" is 99.9% accurate and useless. We undersampled to an 80/20 benign-to-malignant mix, used class weights, and judged models on AUROC as well as accuracy.

**Approach:**

1. **Tabular model:** XGBoost on cleaned and normalized metadata (one-hot encoded sex and body location, min-max scaled numeric features). Trained with 5-fold `StratifiedGroupKFold` grouped by **patient ID**, so no patient appears in both training and validation.
2. **Image model:** fine-tuned Vision Transformer (`google/vit-base-patch16-224-in21k`, Hugging Face `Trainer`) on the lesion images, with color-jitter augmentation to make up for the small number of malignant examples. It outputs a malignancy probability for each lesion.
3. **Fusion model:** XGBoost on the tabular features *plus* the ViT's predicted probabilities. To prevent leakage, the ViT probabilities used as fusion features come from folds where the ViT never saw that lesion (`vit_train_probabilities_no_leakage.csv`).

## Results

| Model | Accuracy |
|---|---|
| ViT, images only (augmented, 20 epochs) | 87.8% |
| Tabular XGBoost, cross-validated by patient | 90.0% |
| **Fusion: tabular + leakage-free ViT probabilities** | **92.99%** (AUROC **0.8907**) |

The fusion model was the most accurate of the three. Adding the image model's predictions to the metadata improved on both single-source models, which confirms the images carry signal the metadata alone does not.

## Conclusions

- **Fusion beats either source alone.** The images and the metadata contribute complementary information, and the combined model was the strongest.
- **Leakage control matters.** Grouping folds by patient and generating out-of-fold ViT probabilities prevents the model from memorizing patients. That gives a more honest, if lower, estimate of real-world performance.
- **Class imbalance shapes everything.** Sampling strategy and the choice of metric had as much impact as model architecture.
- **Next steps:** compute AUROC and partial AUROC (the competition metric) from predicted probabilities, tune XGBoost hyperparameters, and try hair-removal preprocessing, since hair was the most common visual artifact found during data exploration.

## Repository Guide

| File | Purpose |
|---|---|
| `Data_Exploration-Class_FINAL.ipynb` | Loads images from HDF5 and the metadata; explores distributions and class imbalance |
| `Tabular_Data.ipynb` | Metadata cleaning and 5-fold patient-grouped XGBoost; saves out-of-fold probabilities |
| `ViT.ipynb` | Fine-tunes the Vision Transformer with patient-grouped folds |
| `ViT_Augmented_full_run.ipynb` | Full ViT training run with color-jitter augmentation |
| `Tabular_Fusion.ipynb` | Fusion model (tabular + ViT probabilities) compared with tabular-only |
| `fusion_model.ipynb` | Additional fusion experiment on the saved fold probabilities |
| `folds.csv`, `id.csv` | Fold assignments and lesion IDs |
| `vit_*_probabilities*.csv`, `best_tabular_fold_probabilities*.csv` | Saved model outputs used as fusion inputs |

The raw ISIC metadata (`train_metadata.csv`, `test_metadata.csv`) and images are not included, because of their size and the dataset's license. Download them from the [competition page](https://www.kaggle.com/competitions/isic-2024-challenge/data) and place them next to the notebooks.

**Tech:** Python, PyTorch, Hugging Face Transformers, XGBoost, scikit-learn, pandas, h5py, OpenCV, Matplotlib, Seaborn
