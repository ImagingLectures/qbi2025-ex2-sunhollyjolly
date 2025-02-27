from typing import Callable
import numpy as np
from typing import Tuple, Callable
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

'''
def create_dataset_subset(data: np.ndarray, labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.to_numpy()
    if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
        labels = labels.to_numpy()

    indices = np.random.choice(len(data), n, replace=False)
    return data[indices], labels[indices]
'''

def create_dataset_subset(data: np.ndarray, labels: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a balanced subset of n samples from the dataset, ensuring class balance.
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.to_numpy()
    if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    
    # Ensure a balanced subset of labels (0 and 1)
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]
    
    n_per_class = n // 2  # Ensuring equal samples from both classes
    selected_0 = np.random.choice(class_0_indices, n_per_class, replace=False)
    selected_1 = np.random.choice(class_1_indices, n_per_class, replace=False)
    
    indices = np.concatenate((selected_0, selected_1))
    np.random.shuffle(indices)
    
    return data[indices], labels[indices]

def augment_data(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    transform = A.Compose([
        #A.RandomCrop(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.3)
        #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        #A.GaussianBlur(blur_limit=(3, 3), p=0.1),
        #A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize images
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5,border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ])
    
    augmented_imgs = []
    augmented_labels = []
    for img, label in zip(data, labels):
        for _ in range(5):
            transformed = transform(image=img)
            augmented_imgs.append(transformed['image'])
            augmented_labels.append(label)
    return np.array(augmented_imgs), np.array(augmented_labels)

def split_train_test_dataset(data: np.ndarray, labels: np.ndarray, percentage: float = 0.8, shuffle: bool = True) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    return train_test_split(data, labels, test_size=1-percentage, shuffle=shuffle, random_state=42)

def train_kNN(data: np.ndarray, labels: np.ndarray, k: int) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data.reshape(data.shape[0], -1), labels)
    return knn

def predict_kNN(kNN: KNeighborsClassifier, data: np.ndarray) -> np.ndarray:
    return kNN.predict(data.reshape(data.shape[0], -1))

def evaluate_predictions(ground_truth: np.ndarray, labels: np.ndarray, metric: Callable) -> float:
    return metric(ground_truth, labels)

def plot_confusion_matrix(ground_truth: np.ndarray, predictions: np.ndarray, class_names: list):
    cm = confusion_matrix(ground_truth, predictions)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
