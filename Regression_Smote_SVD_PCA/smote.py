from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def euclid_pairwise_dist(x: np.ndarray, y: np.ndarray) ->np.ndarray:
    """
    You implemented this in project 2! We'll give it to you here to save you the copypaste.
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
    """
    x_norm = np.sum(x ** 2, axis=1, keepdims=True)
    yt = y.T
    y_norm = np.sum(yt ** 2, axis=0, keepdims=True)
    dist2 = np.abs(x_norm + y_norm - 2.0 * (x @ yt))
    return np.sqrt(dist2)


def confusion_matrix_vis(conf_matrix: np.ndarray):
    """
    Fancy print of confusion matrix. Just encapsulating some code out of the notebook.
    """
    _, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix)
    ax.set_xlabel('Predicted Labels', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual Labels', fontsize=16)
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, str(val), ha='center', va='center', bbox=dict(
            boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.show()
    return


class SMOTE(object):

    def __init__(self):
        pass

    @staticmethod
    def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray
        ) ->np.ndarray:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            conf_matrix[true, pred] += 1
            
        return conf_matrix



    @staticmethod
    def f1_scores(conf_matrix: np.ndarray) ->np.ndarray:
        f1_scores = []
        
        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            fn = conf_matrix[i, :].sum() - tp
            fp = conf_matrix[:, i].sum() - tp
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            f1_scores.append(f1)
        
        return np.array(f1_scores)

    @staticmethod
    def interpolate(start: np.ndarray, end: np.ndarray, inter_coeff: float
        ) ->np.ndarray:
        return start + inter_coeff * (end - start)

    @staticmethod
    def k_nearest_neighbors(points: np.ndarray, k: int) ->np.ndarray:
        distances = euclid_pairwise_dist(points, points)
        np.fill_diagonal(distances, np.inf) 
        return np.argsort(distances, axis=1)[:, :k]

    @staticmethod
    def smote(X: np.ndarray, y: np.ndarray, k: int, inter_coeff_range:
        Tuple[float]) ->np.ndarray:
        minority_class = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
        majority_class = 1 - minority_class
        num_synthetic = np.sum(y == majority_class) - np.sum(y == minority_class)
        
        minority_points = X[y == minority_class]
        neighbors = SMOTE.k_nearest_neighbors(minority_points, k)
        
        synthetic_points = [
            SMOTE.interpolate(
                start=minority_points[idx],
                end=minority_points[np.random.choice(neighbors[idx])],
                inter_coeff=np.random.uniform(*inter_coeff_range)
            )
            for idx in np.random.choice(len(minority_points), num_synthetic)
        ]
        
        synthetic_X = np.array(synthetic_points)
        synthetic_y = np.full(len(synthetic_points), minority_class)
        
        return synthetic_X, synthetic_y
