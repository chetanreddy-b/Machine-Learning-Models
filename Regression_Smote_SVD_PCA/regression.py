from typing import List, Tuple
import numpy as np

class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:
        mse = np.mean((pred - label) ** 2)
        return np.sqrt(mse)
        
    def construct_polynomial_feats(self, x: np.ndarray, degree: int) -> np.ndarray:
        if x.ndim == 1:
            N = x.shape[0]
            feat = np.zeros((N, degree + 1))
            feat[:, 0] = 1.0  
            for d in range(1, degree + 1):
                feat[:, d] = x ** d
            return feat
        else:
            N, D = x.shape
            feat = np.zeros((N, degree + 1, D))
            feat[:, 0, :] = 1.0 
            for d in range(1, degree + 1):
                feat[:, d, :] = x ** d
            return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return np.dot(xtest, weight)

    def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(xtrain) @ ytrain

    def linear_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs: int = 5, learning_rate: float = 0.001) -> Tuple[np.ndarray, List[float]]:
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_epoch = []

        for _ in range(epochs):
            prediction = self.predict(xtrain, weight)
            error = prediction - ytrain
            weight -= (learning_rate / N) * np.dot(xtrain.T, error)
            prediction_updated = self.predict(xtrain, weight)
            loss = self.rmse(prediction_updated, ytrain)
            loss_per_epoch.append(loss)

        return weight, loss_per_epoch

    def linear_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs: int = 100, learning_rate: float = 0.001) -> Tuple[np.ndarray, List[float]]:
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_step = []

        for _ in range(epochs):
            for i in range(N):
                xi = xtrain[i, :].reshape(1, -1)
                yi = ytrain[i]
                prediction = np.dot(xi, weight)
                error = prediction - yi
                weight -= learning_rate * (xi.T * error)
                loss = self.rmse(self.predict(xtrain, weight), ytrain)
                loss_per_step.append(loss)

        return weight, loss_per_step

    def ridge_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float) -> np.ndarray:
        N, D = xtrain.shape
        I = np.eye(D)
        I[0, 0] = 0 
        return np.linalg.pinv(xtrain.T @ xtrain + c_lambda * I) @ xtrain.T @ ytrain

    def ridge_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float, epochs: int = 500, learning_rate: float = 1e-07) -> Tuple[np.ndarray, List[float]]:
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_epoch = []

        for _ in range(epochs):
            prediction = self.predict(xtrain, weight)
            error = prediction - ytrain
            regularization_term = (c_lambda / N) * weight
            regularization_term[0] = 0 
            weight -= learning_rate * ((1 / N) * np.dot(xtrain.T, error) + regularization_term)
            prediction_updated = self.predict(xtrain, weight)
            loss = self.rmse(prediction_updated, ytrain)
            loss_per_epoch.append(loss)

        return weight, loss_per_epoch

    def ridge_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float, epochs: int = 100, learning_rate: float = 0.001) -> Tuple[np.ndarray, List[float]]:
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_step = []

        for _ in range(epochs):
            for i in range(N):
                xi = xtrain[i, :].reshape(1, -1)
                yi = ytrain[i]
                prediction = np.dot(xi, weight)
                error = prediction - yi
                regularization_term = (c_lambda / N) * weight
                regularization_term[0] = 0 
                weight -= learning_rate * (xi.T * error + regularization_term)
                loss = self.rmse(self.predict(xtrain, weight), ytrain)
                loss_per_step.append(loss)

        return weight, loss_per_step

    def ridge_cross_validation(self, X: np.ndarray, y: np.ndarray, kfold: int = 5, c_lambda: float = 100) -> List[float]:
        fold_size = X.shape[0] // kfold
        loss_per_fold = []

        for i in range(kfold):
            X_train = np.concatenate([X[:i * fold_size, :], X[(i + 1) * fold_size:, :]], axis=0)
            y_train = np.concatenate([y[:i * fold_size, :], y[(i + 1) * fold_size:, :]], axis=0)
            X_val = X[i * fold_size:(i + 1) * fold_size, :]
            y_val = y[i * fold_size:(i + 1) * fold_size, :]

            weight = self.ridge_fit_closed(X_train, y_train, c_lambda)
            pred = self.predict(X_val, weight)
            loss = self.rmse(pred, y_val)
            loss_per_fold.append(loss)

        return loss_per_fold


    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
        lambda_list: List[float], kfold: int) ->Tuple[float, float, List[float]
        ]:
        best_error = None
        best_lambda = None
        error_list = []
        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm
        return best_lambda, best_error, error_list
