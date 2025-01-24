from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


class LogisticRegression(object):

    def __init__(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.epoch_list = []

    def sigmoid(self, s: np.ndarray) ->np.ndarray:
        return 1 / (1 + np.exp(-s))
        raise NotImplementedError

    def bias_augment(self, x: np.ndarray) ->np.ndarray:
        return np.hstack((np.ones((x.shape[0], 1)), x))

        raise NotImplementedError

    def predict_probs(self, x_aug: np.ndarray, theta: np.ndarray) ->np.ndarray:
        return self.sigmoid(x_aug @ theta)

        raise NotImplementedError

    def predict_labels(self, h_x: np.ndarray, thresold: float) ->np.ndarray:
        return (h_x >= thresold).astype(int)


        raise NotImplementedError

    def loss(self, y: np.ndarray, h_x: np.ndarray) ->float:
        N = y.shape[0]
        return -np.mean(y * np.log(h_x + 1e-10) + (1 - y) * np.log(1 - h_x + 1e-10))
        raise NotImplementedError

    def gradient(self, x_aug: np.ndarray, y: np.ndarray, h_x: np.ndarray
        ) ->np.ndarray:
        N = x_aug.shape[0]
        return (x_aug.T @ (h_x - y)) / N
        raise NotImplementedError

    def accuracy(self, y: np.ndarray, y_hat: np.ndarray) ->float:
        return np.mean(y == y_hat)
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray,
        threshold: float) ->Tuple[float, float]:
        x_aug = self.bias_augment(x)
        h_x = self.predict_probs(x_aug, theta)
        y_hat = self.predict_labels(h_x, threshold)
        
        loss = self.loss(y, h_x + 1e-10)
        acc = self.accuracy(y, y_hat)
    
        return loss, acc

        raise NotImplementedError

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.
        ndarray, y_val: np.ndarray, lr: float, epochs: int, threshold: float
        ) ->Tuple[np.ndarray, List[float], List[float], List[float], List[
        float], List[int]]:
        N, D = x_train.shape
        theta = np.zeros((D + 1, 1))

        x_train_aug = self.bias_augment(x_train)

        for epoch in range(epochs):
            h_x = self.predict_probs(x_train_aug, theta)            
            grad = self.gradient(x_train_aug, y_train, h_x)            
            theta -= lr * grad            
            if epoch % 100 == 0:
                self.update_evaluation_lists(x_train, y_train, x_val, y_val, theta, epoch, threshold)

        return theta

        raise NotImplementedError

    def update_evaluation_lists(self, x_train: np.ndarray, y_train: np.
        ndarray, x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray,
        epoch: int, threshold: float):
        train_loss, train_acc = self.evaluate(x_train, y_train, theta, threshold)
        val_loss, val_acc = self.evaluate(x_val, y_val, theta, threshold)
        self.epoch_list.append(epoch)
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        if epoch % 1000 == 0:
            print(
                f"""Epoch {epoch}:
	train loss: {round(train_loss, 3)}	train acc: {round(train_acc, 3)}
	val loss:   {round(val_loss, 3)}	val acc:   {round(val_acc, 3)}"""
                )

    def plot_loss(self, train_loss_list: List[float]=None, val_loss_list:
        List[float]=None, epoch_list: List[int]=None) ->None:
        if train_loss_list is None:
            assert hasattr(self, 'train_loss_list')
            assert hasattr(self, 'val_loss_list')
            assert hasattr(self, 'epoch_list')
            train_loss_list = self.train_loss_list
            val_loss_list = self.val_loss_list
            epoch_list = self.epoch_list
        fig = px.line(x=epoch_list, y=[train_loss_list, val_loss_list],
            labels={'x': 'Epoch', 'y': 'Loss'}, title='Loss')
        labels = ['Train Loss', 'Validation Loss']
        for idx, trace in enumerate(fig['data']):
            trace['name'] = labels[idx]
        fig.update_layout(legend_title_text='Loss Type')
        fig.show()

    def plot_accuracy(self, train_acc_list: List[float]=None, val_acc_list:
        List[float]=None, epoch_list: List[int]=None) ->None:
        if train_acc_list is None:
            assert hasattr(self, 'train_acc_list')
            assert hasattr(self, 'val_acc_list')
            assert hasattr(self, 'epoch_list')
            train_acc_list = self.train_acc_list
            val_acc_list = self.val_acc_list
            epoch_list = self.epoch_list
        fig = px.line(x=epoch_list, y=[train_acc_list, val_acc_list],
            labels={'x': 'Epoch', 'y': 'Accuracy'}, title='Accuracy')
        labels = ['Train Accuracy', 'Validation Accuracy']
        for idx, trace in enumerate(fig['data']):
            trace['name'] = labels[idx]
        fig.update_layout(legend_title_text='Accuracy Type')
        fig.show()