import numpy as np
import pandas as pd
import plotly.express as px

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X.
        
        Args:
            X: (N,D) numpy array corresponding to a dataset

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        X_centered = X - np.mean(X, axis=0)        
        self.U, self.S, self.V = np.linalg.svd(X_centered, full_matrices=False)

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns).
        
        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        data_centered = data - np.mean(data, axis=0)        
        X_new = np.dot(data_centered, self.V[:K].T)
        return X_new

    def transform_rv(self, data: np.ndarray, retained_variance: float = 0.99) -> np.ndarray:
        """
        Transform data to reduce the number of features such that the retained variance is kept in X_new with K features.
        
        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        cumulative_variance = np.cumsum(self.S ** 2) / np.sum(self.S ** 2)        
        K = np.searchsorted(cumulative_variance, retained_variance) + 1
        data_centered = data - np.mean(data, axis=0)        
        X_new = np.dot(data_centered, self.V[:K].T)
        return X_new

    def get_V(self) -> np.ndarray:
        """
        Getter function for value of V.
        """
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:
        """
        You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
        
        Args:
            X: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            y: (N,) numpy array, the true labels

        Return: None
        """
        self.fit(X)        
        X_reduced_2D = self.transform(X, 2)
        X_reduced_df = pd.DataFrame(X_reduced_2D, columns=['PC1', 'PC2'])
        X_reduced_df['Label'] = y
        fig_2D = px.scatter(X_reduced_df, x='PC1', y='PC2', color='Label', title=f'{fig_title}: 2D Plot of Principal Components')
        fig_2D.update_layout(width=600, height=600).show()
        fig_3D = px.scatter_3d(X_reduced_df, x='PC1', y='PC2', z='PC1', color='Label', title=f'{fig_title}: 3D Plot of Principal Components')
        fig_3D.update_layout(width=600, height=600).show()
        random_features = np.random.choice(X.shape[1], 2, replace=False)
        X_random_df = pd.DataFrame(X[:, random_features], columns=['Feature1', 'Feature2'])
        X_random_df['Label'] = y
        fig_random_2D = px.scatter(X_random_df, x='Feature1', y='Feature2', color='Label', title=f'{fig_title}: 2D Plot of Random Features')
        fig_random_2D.update_layout(width=600, height=600).show()
