"""
Hyperspectral Image Classification using 2D Spatial-Spectral Convolutional Neural Networks
Dataset: Salinas Valley
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import warnings
warnings.filterwarnings("ignore")
K.set_image_data_format('channels_last')

def load_data(data_path, labels_path):
    """Loads the Salinas dataset from .mat files."""
    data = sio.loadmat(data_path)['salinas_corrected']
    labels = sio.loadmat(labels_path)['salinas_gt']
    return data, labels

def splitTrainTestSet(X, y, testRatio=0.25):
    """Splits the data into training and testing sets stratifying by class."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testRatio, random_state=345, stratify=y
    )
    return X_train, X_test, y_train, y_test

def oversampleWeakClasses(X, y):
    """Balances the dataset by oversampling classes with fewer samples."""
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts

    newX = X[y == uniqueLabels[0], :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)

    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y == label,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))

    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :]
    newY = newY[rand_perm]
    return newX, newY

def applyPCA(X, numComponents=75):
    """Applies Principal Component Analysis to reduce spectral dimensions."""
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    """Pads the image with zeros to allow patch extraction at the borders."""
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    """Extracts spatial-spectral patches (cubes) from the HSI."""
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0

    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def visualize_results(y_true, y_pred):
    """Creates a professional side-by-side plot of Ground Truth vs Prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    cmap = 'nipy_spectral'

    axes[0].imshow(y_true, cmap=cmap)
    axes[0].set_title('Ground Truth Map', fontsize=14)
    axes[0].axis('off')

    im = axes[1].imshow(y_pred, cmap=cmap)
    axes[1].set_title('2D Spatial-Spectral Prediction Map', fontsize=14)
    axes[1].axis('off')

    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    plt.suptitle('Salinas Hyperspectral Image Classification', fontsize=18)
    plt.show()

def main(data_path, labels_path):
    numComponents = 30
    testRatio = 0.25
    windowSize = 11

    print(" Loading data...")
    X, y = load_data(data_path, labels_path)
    height, width, bands = X.shape
    print(f"Original Image Shape: {height}x{width} with {bands} bands.")

    print(f"Applying PCA to reduce to {numComponents} components...")
    X_pca, pca = applyPCA(X, numComponents=numComponents)

    print(f" Extracting {windowSize}x{windowSize} spatial patches and balancing data...")
    X_patches, y_patches = createImageCubes(X_pca, y, windowSize=windowSize)

    X_train, X_test, y_train, y_test = splitTrainTestSet(X_patches, y_patches, testRatio)
    X_train, y_train = oversampleWeakClasses(X_train, y_train)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print("Building and compiling 2D CNN model...")
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(windowSize, windowSize, numComponents)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='softmax')) # 16 classes in Salinas

    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    # 8. Train Model
    print(" Training model...")
    model.fit(X_train, y_train, batch_size=64, epochs=15, verbose=1, validation_data=(X_test, y_test))

    print("Predicting entire image mapping...")
    X_all_patches, _ = createImageCubes(X_pca, y, windowSize=windowSize, removeZeroLabels=False)

    predictions = model.predict(X_all_patches, batch_size=512)

    predicted_classes = np.argmax(predictions, axis=1) + 1

    outputs = predicted_classes.reshape(height, width)

    outputs[y == 0] = 0

    print("[INFO] Generating visualization...")
    visualize_results(y, outputs)


if __name__ == "__main__":

    DATA_PATH = "salinas_corrected.mat"
    LABELS_PATH = "salinas_gt.mat"
    main(DATA_PATH, LABELS_PATH)

