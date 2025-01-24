import numpy as np

class KMeans(object):
    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        self.centers = self.points[np.random.choice(self.points.shape[0], self.K, replace=False)]
        return self.centers

    def kmpp_init(self):
        sample_size = max(1, int(0.01 * self.points.shape[0]))
        sampled_data = self.points[np.random.choice(self.points.shape[0], sample_size, replace=False)]
        centers = [sampled_data[np.random.choice(sampled_data.shape[0])]]
        for _ in range(1, self.K):
            distances = np.min([np.sum((sampled_data - center) ** 2, axis=1) for center in centers], axis=0)
            next_center = sampled_data[np.argmax(distances)]
            centers.append(next_center)
        self.centers = np.vstack(centers)
        return self.centers

    def update_assignment(self):
        distances = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(distances, axis=1)
        return self.assignments

    def update_centers(self):
        updated_centers = []
        for k in range(self.K):
            cluster_points = self.points[self.assignments == k]
            if len(cluster_points) > 0:
                updated_centers.append(cluster_points.mean(axis=0))
            else:
                updated_centers.append(self.points[np.random.choice(self.points.shape[0])])
        self.centers = np.array(updated_centers)
        return self.centers

    def get_loss(self):
        distances = pairwise_dist(self.points, self.centers)
        squared_distances = np.sum((distances[np.arange(len(self.points)), self.assignments]) ** 2)
        self.loss = np.sum(squared_distances)
        return self.loss

    def train(self):
        prev_loss = float("inf")
        for _ in range(self.max_iters):
            self.update_assignment()
            self.update_centers()
            for k in range(self.K):
                if not np.any(self.assignments == k):
                    self.centers[k] = self.points[np.random.choice(self.points.shape[0])]
            self.get_loss()
            if abs(prev_loss - self.loss) / prev_loss < self.rel_tol:
                break
            prev_loss = self.loss
        return self.centers, self.assignments, self.loss

def pairwise_dist(x, y):
    x_sq = np.sum(x ** 2, axis=1).reshape(-1, 1)
    y_sq = np.sum(y ** 2, axis=1).reshape(1, -1)
    xy_product = np.dot(x, y.T)
    dist_sq = x_sq + y_sq - 2 * xy_product
    dist = np.sqrt(dist_sq)
    dist = np.maximum(dist, 0)
    return dist

def fowlkes_mallow(x_ground_truth, x_predicted):
    N = len(x_ground_truth)
    true_positives, false_positives, false_negatives = 0, 0, 0
    for i in range(N):
        for j in range(i + 1, N):
            same_pred = x_predicted[i] == x_predicted[j]
            same_gt = x_ground_truth[i] == x_ground_truth[j]
            if same_pred and same_gt:
                true_positives += 1
            elif same_pred and not same_gt:
                false_positives += 1
            elif not same_pred and same_gt:
                false_negatives += 1
    denominator = np.sqrt((true_positives + false_positives) * (true_positives + false_negatives))
    return 0.0 if denominator == 0 else float(true_positives / denominator)
