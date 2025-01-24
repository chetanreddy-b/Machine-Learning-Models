import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32
FULL_MATRIX = True

class GMM:
    def __init__(self, X, K, max_iters=100):
        self.points = X
        self.N, self.D = X.shape
        self.K = K
        self.max_iters = max_iters

    def softmax(self, logit):
        max_logit = np.max(logit, axis=1, keepdims=True)
        stable_logit = logit - max_logit
        exp_vals = np.exp(stable_logit)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def logsumexp(self, logit):
        max_val = np.max(logit, axis=1, keepdims=True)
        exp_shifted = np.exp(logit - max_val)
        return max_val + np.log(np.sum(exp_shifted, axis=1, keepdims=True))

    def normalPDF(self, points, mu_i, sigma_i):
        diag_sigma = np.diagonal(sigma_i)
        factor = 1.0 / np.sqrt((2 * np.pi) ** mu_i.shape[0] * np.prod(diag_sigma))
        diff = points - mu_i
        exponent_term = -0.5 * np.sum((diff ** 2) / diag_sigma, axis=1)
        return factor * np.exp(exponent_term)

    def multinormalPDF(self, points, mu_i, sigma_i):
        try:
            inv_sigma = np.linalg.inv(sigma_i)
        except LinAlgError:
            inv_sigma = np.linalg.inv(sigma_i + SIGMA_CONST * np.eye(mu_i.shape[0]))
        det_sigma = np.linalg.det(sigma_i)
        det_sigma = det_sigma if det_sigma != 0 else SIGMA_CONST
        norm_factor = 1.0 / np.sqrt((2 * np.pi) ** mu_i.shape[0] * det_sigma)
        diff = points - mu_i
        exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
        return norm_factor * np.exp(exponent)

    def create_pi(self):
        return np.full(self.K, 1 / self.K)

    def create_mu(self):
        return self.points[np.random.choice(self.N, self.K, replace=True)]

    def create_sigma(self):
        return np.array([np.eye(self.D) for _ in range(self.K)])

    def _init_components(self):
        np.random.seed(5)
        pi = self.create_pi()
        mu = self.create_mu()
        sigma = self.create_sigma()
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX):
        log_likelihoods = np.zeros((self.N, self.K))
        for k in range(self.K):
            if full_matrix:
                pdf_vals = self.multinormalPDF(self.points, mu[k], sigma[k])
            else:
                pdf_vals = self.normalPDF(self.points, mu[k], sigma[k])
            log_likelihoods[:, k] = np.log(pi[k] + LOG_CONST) + np.log(pdf_vals + LOG_CONST)
        return log_likelihoods

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX):
        joint_log_likelihood = self._ll_joint(pi, mu, sigma, full_matrix=full_matrix)
        return self.softmax(joint_log_likelihood)

    def _M_step(self, tau, full_matrix=FULL_MATRIX):
        weighted_points = np.sum(tau, axis=0)
        pi = weighted_points / self.N
        mu = (tau.T @ self.points) / weighted_points[:, np.newaxis]
        sigma = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            diff = self.points - mu[k]
            if full_matrix:
                sigma[k] = (tau[:, k][:, np.newaxis] * diff).T @ diff / weighted_points[k]
            else:
                diag_vals = np.sum(tau[:, k][:, np.newaxis] * diff ** 2, axis=0) / weighted_points[k]
                sigma[k] = np.diag(diag_vals)
        return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16):
        pi, mu, sigma = self._init_components()
        pbar = tqdm(range(self.max_iters))
        prev_loss = None
        for iteration in pbar:
            tau = self._E_step(pi, mu, sigma, full_matrix)
            pi, mu, sigma = self._M_step(tau, full_matrix)
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if prev_loss is not None:
                if np.abs(prev_loss - loss) < abs_tol and (np.abs(prev_loss - loss) / prev_loss) < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description(f"iter {iteration}, loss: {loss:.4f}")
        return tau, (pi, mu, sigma)

def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
    reshaped_img = image.reshape(-1, 3).astype(np.float32)
    gmm_model = GMM(reshaped_img, K, max_iters=max_iters)
    tau, (pi, mu, sigma) = gmm_model()
    pixel_labels = np.argmax(tau, axis=1)
    clustered_pixels = mu[pixel_labels].clip(0, 255).astype(np.uint8)
    return clustered_pixels.reshape(image.shape)
