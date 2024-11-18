from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


#Preprocesses data by loading and splitting the data.
"""
    Preprocesses the dataset by loading, splitting into train/validation/test sets, and returning these sets.

    Parameters:
    -----------
    n : int
        Dataset version identifier.
    validate : bool, optional (default=True)
        If True, includes a validation set; otherwise, only train and test sets are created.

    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : ndarray
        Arrays of features and labels for training, validation, and testing.
"""

def preprocess(n: int, validate: bool = True, standardize: bool = False):
    X = np.load(f'../Datasets/kryptonite-{n}-X.npy')
    y = np.load(f'../Datasets/kryptonite-{n}-y.npy')

    if validate:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        X_val, y_val = None, None

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if validate:
            X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def targetPerformance(n: int) -> float:
    target_dict = {
        9: 0.95,
        12: 0.925,
        15: 0.90,
        18: 0.875,
        24: 0.80,
        30: 0.75,
        45: 0.70
    }
    return target_dict.get(n, None)  # n이 없을 경우 None을 반환




def plot_training_curves(training_losses, validation_losses, validation_accuracies):
    epochs = range(1, len(training_losses) + 1)
    
    plt.figure(figsize=(14, 5))

    # 학습 손실 및 검증 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, 'b', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 검증 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_accuracies, 'g', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
