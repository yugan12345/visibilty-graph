# Image Classification using IHVG and Local Motifs

This repository implements a hybrid image classification pipeline based on the paper **"Image Classification Using Visibility Graphs and Local Binary Patterns" (IEEE Access, 2018)**. The method combines global features derived from Image Horizontal Visibility Graphs (IHVG) and local features extracted using 3x3 patch-based visibility motifs (similar to LBP).

We further introduce a novelty in our research by attempting to integrate **small-world network properties** in the generated graphs to explore potential improvements in classification accuracy.

## Pipeline Overview

1. **Image Loading**: Grayscale images are loaded from subfolders (one per class).
2. **Resizing**: All images are resized to 160×160.
3. **Feature Extraction**:
   - **Global Features**: IHVG degree-based histogram (32 bins).
   - **Local Features**: Patch-based 3x3 visibility motifs, encoded into a 256-length vector.
4. **Feature Vector Construction**: Global and local features are concatenated.
5. **Normalization & Dimensionality Reduction**:
   - Features are scaled using `StandardScaler`.
   - PCA is applied (retain 95% variance).
6. **Training**:
   - Data is split into training/testing sets (stratified).
   - An SVM with RBF kernel is trained on the PCA-transformed features.
7. **Evaluation**:
   - Accuracy is printed.
   - A grid of 10 test images with predicted vs. true labels is visualized.

## Novelty Contribution

We experimented with the incorporation of **small-world properties** in the IHVG graphs to enhance the model's capability to capture global texture dependencies. This experimental variation explores whether introducing long-range visibility edges leads to any boost in classification performance. The idea is based on known advantages of small-world networks in biological and information systems.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Install dependencies:

```bash
pip install numpy opencv-python matplotlib scikit-learn
