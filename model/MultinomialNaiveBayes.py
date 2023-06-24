import numpy as np
import time


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.features = X.shape[1]
        self.class_counts = np.zeros(self.classes.shape[0])
        self.log_class_prior = np.zeros(self.classes.shape[0])
        self.log_likelihoods = np.zeros((self.classes.shape[0], self.features))

        for c in self.classes:
            c -= 1
            X_c = X[c == y]
            self.class_counts[c] = X_c.shape[0]
            self.log_class_prior[c] = np.log(self.class_counts[c]) - np.log(
                X.shape[0])  # Applying log to prevent underflow
            self.log_likelihoods[c, :] = np.log((X_c.sum(axis=0) + self.alpha) / (
                np.sum(X_c.sum(axis=0) + self.alpha)))  # Applying log to prevent underflow

    def predict_single(self, x):
        log_posteriors = np.zeros(self.classes.shape[0])
        for c in self.classes:
            c -= 1
            log_posteriors[c] = self.log_class_prior[c] + np.sum(x * self.log_likelihoods[c, :])
        return self.classes[np.argmax(log_posteriors)]

    def predict(self, X):
        start_time = time.time()
        result = np.array([self.predict_single(x) for x in X])
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.2f} seconds")
        return result
