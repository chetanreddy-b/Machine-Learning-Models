import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
SIGMA_CONST = 1e-06
LOG_CONST = 1e-32

def complete_(data):
    features = data[:, :-1]
    labels = data[:, -1]
    features_complete = ~np.isnan(features).any(axis=1)
    labels_complete = ~np.isnan(labels)
    mask = features_complete & labels_complete
    return data[mask]

def incomplete_(data):
    features = data[:, :-1]
    labels = data[:, -1]
    features_incomplete = np.isnan(features).any(axis=1)
    labels_complete = ~np.isnan(labels)
    mask = features_incomplete & labels_complete
    return data[mask]

def unlabeled_(data):
    features = data[:, :-1]
    labels = data[:, -1]
    features_complete = ~np.isnan(features).any(axis=1)
    labels_incomplete = np.isnan(labels)
    mask = features_complete & labels_incomplete
    return data[mask]

class CleanData(object):
    def __init__(self):
        pass

    def pairwise_dist(self, x, y):
        x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
        y_norm = np.sum(y**2, axis=1).reshape(1, -1)
        dist_sq = x_norm + y_norm - 2 * np.dot(x, y.T)
        dist_sq = np.maximum(dist_sq, 0)
        return np.sqrt(dist_sq)

    def __call__(self, incomplete_points, complete_points, K, **kwargs):
        clean_points = np.copy(incomplete_points)
        for i, incomplete in enumerate(incomplete_points):
            missing_index = np.where(np.isnan(incomplete[:-1]))[0][0]
            same_class_complete = complete_points[complete_points[:, -1] == incomplete[-1]]
            distances = self.pairwise_dist(incomplete[np.newaxis, :-1], same_class_complete[:, :-1])
            k_nearest_neighbors = np.argsort(distances, axis=1)[:, :K]
            clean_points[i, missing_index] = np.mean(same_class_complete[k_nearest_neighbors.flatten(), missing_index])
        return np.vstack([complete_points, clean_points])

def median_clean_data(data):
    data_copy = np.copy(data)
    for col in range(data.shape[1] - 1):
        col_median = np.nanmedian(data_copy[:, col])
        data_copy[np.isnan(data_copy[:, col]), col] = col_median
    return np.around(data_copy, decimals=1)

class SemiSupervised(object):
    def __init__(self):
        pass

    def softmax(self, logit):
        exp_logit = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        return exp_logit / np.sum(exp_logit, axis=1, keepdims=True)

    def logsumexp(self, logit):
        logit_max = np.max(logit, axis=1, keepdims=True)
        return logit_max + np.log(np.sum(np.exp(logit - logit_max), axis=1, keepdims=True))

    def normalPDF(self, logit, mu_i, sigma_i):
        D = logit.shape[1]
        sigma_inv = np.linalg.inv(sigma_i + SIGMA_CONST * np.eye(D))
        diff = logit - mu_i
        exponent = -0.5 * np.sum(diff @ sigma_inv * diff, axis=1)
        return np.exp(exponent) / (np.sqrt((2 * np.pi) ** D * np.linalg.det(sigma_i + SIGMA_CONST * np.eye(D))))

    def _init_components(self, points, K, **kwargs):
        labeled_points = points[~np.isnan(points[:, -1])]
        N, D_plus_1 = labeled_points.shape
        D = D_plus_1 - 1
        labels = labeled_points[:, -1].astype(int)
        pi = np.bincount(labels, minlength=K) / len(labels)
        mu = np.zeros((K, D))
        for k in range(K):
            class_points = labeled_points[labels == k, :-1]
            if len(class_points) > 0:
                mu[k] = np.mean(class_points, axis=0)
        sigma = np.zeros((K, D, D))
        for k in range(K):
            class_points = labeled_points[labels == k, :-1]
            if len(class_points) > 0:
                sigma[k] = np.diag(np.var(class_points, axis=0) + SIGMA_CONST)
            else:
                sigma[k] = np.eye(D) * SIGMA_CONST
        return pi, mu, sigma

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):
        N = points.shape[0]
        K = pi.shape[0]
        ll = np.zeros((N, K))
        for k in range(K):
            log_pi_k = np.log(pi[k] + LOG_CONST)
            ll[:, k] = log_pi_k + np.log(self.normalPDF(points, mu[k], sigma[k]) + LOG_CONST)
        return ll

    def _E_step(self, points, pi, mu, sigma, **kwargs):
        ll_joint = self._ll_joint(points, pi, mu, sigma)
        return self.softmax(ll_joint)

    def _M_step(self, points, gamma, **kwargs):
        N, D = points.shape
        K = gamma.shape[1]
        N_k = np.sum(gamma, axis=0)
        pi = N_k / N
        mu = np.dot(gamma.T, points) / N_k[:, np.newaxis]
        sigma = np.zeros((K, D, D))
        for k in range(K):
            diff = points - mu[k]
            sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / N_k[k] + SIGMA_CONST * np.eye(D)
        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        N, D_plus_1 = points.shape
        D = D_plus_1 - 1
        data_labeled = points[~np.isnan(points[:, -1])]
        pi, mu, sigma = self._init_components(data_labeled, K, **kwargs)
        for i in range(max_iters):
            gamma = np.zeros((N, K))
            idx_labeled = ~np.isnan(points[:, -1])
            gamma_labeled = np.zeros((np.sum(idx_labeled), K))
            gamma_labeled[np.arange(gamma_labeled.shape[0]), data_labeled[:, -1].astype(int)] = 1
            ll_unlabeled = self._ll_joint(points[np.isnan(points[:, -1]), :-1], pi, mu, sigma, **kwargs)
            gamma_unlabeled = self.softmax(ll_unlabeled)
            gamma[idx_labeled] = gamma_labeled
            gamma[~idx_labeled] = gamma_unlabeled
            pi, mu, sigma = self._M_step(points[:, :-1], gamma, **kwargs)
            ll_joint = self._ll_joint(points[:, :-1], pi, mu, sigma, **kwargs)
            ll = np.sum(self.logsumexp(ll_joint))
            if i > 0 and abs(ll - ll_old) < abs_tol:
                break
            if i > 0 and abs(ll - ll_old) / abs(ll_old) < rel_tol:
                break
            ll_old = ll
        return pi, mu, sigma

class ComparePerformance(object):
    def __init__(self):
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K: int) -> float:
        model = SemiSupervised()
        pi, mu, sigma = model(training_data, K)
        ll_val = model._ll_joint(validation_data[:, :-1], pi, mu, sigma)
        predictions = np.argmax(ll_val, axis=1)
        return accuracy_score(validation_data[:, -1], predictions)

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float:
        training_data = training_data[~np.isnan(training_data).any(axis=1)]
        clf = GaussianNB()
        clf.fit(training_data[:, :-1], training_data[:, -1])
        predictions = clf.predict(validation_data[:, :-1])
        return accuracy_score(validation_data[:, -1], predictions)
