import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import ExtraTreeClassifier

class RandomForest(object):
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=max_depth, criterion="entropy")
            for i in range(n_estimators)
        ]
        self.alphas = (
            []
        )  # Importance values for adaptive boosting extra credit implementation


    def _bootstrapping(self, num_training, num_features, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        row_ind = np.random.choice(num_training, num_training, replace=True)
        col_ind = np.random.choice(num_features, int(self.max_features * num_features), replace=False)

        return row_ind, col_ind
        raise NotImplementedError()


    def bootstrapping(self, num_training, num_features):
        np.random.seed(self.random_seed)
        for _ in range(self.n_estimators):
            total = set(range(num_training))
            row_ind, col_ind = self._bootstrapping(num_training, num_features)
            total -= set(row_ind)
            self.bootstraps_row_indices.append(row_ind)
            self.feature_indices.append(col_ind)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        num_training, num_features = X.shape
        self.bootstrapping(num_training, num_features)

        for i in range(self.n_estimators):
            row_ind = self.bootstraps_row_indices[i]
            col_ind = self.feature_indices[i]
            self.decision_trees[i].fit(X[row_ind][:, col_ind], y[row_ind])

    def adaboost(self, X, y):
        N = X.shape[0]
        weights = np.ones(N) / N

        for t, tree in enumerate(self.decision_trees):
            tree.fit(X, y, sample_weight=weights)

            incorrect = (tree.predict(X) != y)
            weighted_error = np.sum(weights * incorrect) / np.sum(weights)
            alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            self.alphas.append(alpha)

            weights *= np.exp(alpha * incorrect * 2)
            weights /= np.sum(weights)  

    def OOB_score(self, X, y):
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if predictions:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros((N, 7))
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        pred = np.argmax(y, axis=1)
        return pred

    def predict_adaboost(self, X):
        # Helper method. You don't have to modify it.
        # This function makes predictions using AdaBoost ensemble by aggregating weighted votes.
        N = X.shape[0]
        weighted_votes = np.zeros((N, 7))

        for alpha, tree in zip(self.alphas, self.decision_trees[: len(self.alphas)]):
            pred = tree.predict(X)
            for i in range(N):
                class_index = int(pred[i])
                weighted_votes[i, class_index] += alpha

        return np.argmax(weighted_votes, axis=1)
    
    def plot_feature_importance(self, data_train):
        tree = self.decision_trees[0]
        feature_importances = tree.feature_importances_
        selected_features = self.feature_indices[0]
        feature_col_names = data_train.columns[selected_features]

        if len(feature_importances) != len(selected_features):
            raise ValueError("Mismatch b/w feature importances & selected features.")
        
        non_zero_indices = [i for i, importance in enumerate(feature_importances) if importance > 0]
        feature_importances = [feature_importances[i] for i in non_zero_indices]
        feature_col_names = [feature_col_names[i] for i in non_zero_indices]
        indices_sort = sorted(range(len(feature_importances)), key=lambda i: feature_importances[i], reverse=True)
        feature_importances = [feature_importances[i] for i in indices_sort]
        feature_col_names = [feature_col_names[i] for i in indices_sort]

        plt.figure(figsize=(10, 6))
        plt.bar(feature_col_names,feature_importances,color='red')
        plt.ylabel("Feature Importance")
        plt.xlabel("Features")
        plt.title("Feature Importance of selected features in Decision Tree")
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def select_hyperparameters(self):
        """
        Hyperparameter tuning Question
        Assign values to n_estimators, max_depth, max_features
        """
        n_estimators = 14      
        max_depth = 10          
        max_features = 0.7     
        
        return n_estimators, max_depth, max_features
