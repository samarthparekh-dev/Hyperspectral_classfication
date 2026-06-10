# Hyperspectral Image Classification using 2D Spatial-Spectral Convolutional Neural Networks

This repository contains a deep learning pipeline optimized for Hyperspectral Image (HSI) land-cover classification using the benchmark Salinas Valley dataset.

Using a 2D Convolutional Neural Network (CNN), this project demonstrates how combining spectral information (dimensionality-reduced via PCA) with spatial context ($11 \times 11$ neighboring pixel neighborhoods) resolves the common "salt-and-pepper" misclassification noise associated with standard 1D/pixel-by-pixel classifiers.

Project Overview :

Hyperspectral sensors capture hundreds of contiguous spectral bands, providing rich information but also introducing the curse of dimensionality (highly correlated data) and spatial noise.

This project solves these challenges using a robust, multi-step pipeline:

Dimensionality Reduction: Compresses the original 224 spectral bands to 30 principal components using PCA, retaining over 99% of the variance while drastically lowering computation time.

Spatial-Spectral Patching: Extracts spatial cubes ($11 \times 11 \times 30$) around each target pixel to give the CNN neighborhood context.

Imbalance Mitigation: Addresses extreme class distribution issues using a custom oversampling function for minority agricultural classes.

Deep Learning Classification: Trains a 2D CNN with regularization (Dropout) to learn spatial features (edges, textures, field boundaries) alongside spectral signatures.

High-Speed Map Inference: Performs vectorized block prediction to reconstruct a smooth, highly accurate classification map of the entire valley floor.

📊 Dataset: Salinas Valley

The Salinas corrected dataset was gathered by the AVIRIS sensor over Salinas Valley, California.

Spatial Resolution: $512 \times 217$ pixels

Spectral Resolution: 224 bands (reduced to 30 components in this project)

Classes: 16 agricultural land-cover types (e.g., Broccoli, Grapes, Lettuce, Vinyard, etc.) and 1 background class (unlabeled).

RESULTS : 
<img width="884" height="652" alt="download (5)" src="https://github.com/user-attachments/assets/3c7d1ac2-fefd-4d82-9563-e5e7526820c1" />
