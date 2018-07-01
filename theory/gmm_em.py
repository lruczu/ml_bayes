import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance


class GMM:
    def __init__(self, n_clusters, n_iter=100, tol=0.00001):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X):
        n, n_dim = X.shape
        km = KMeans(n_clusters=self.n_clusters)
        X_init = km.fit_transform(X)

        mean = np.zeros((self.n_clusters, n_dim))
        std = np.zeros((self.n_clusters, n_dim, n_dim))
        weights = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            mean[i] = X[km.labels_ == i].mean(axis=0)[np.newaxis, :]
            std[i] = empirical_covariance(X[km.labels_ == i])
            weights[i] = (km.labels_ == i).mean()

        q = np.ones((n, self.n_clusters))
        q = q / q.sum(axis=1)[:, np.newaxis]

        L_prev = -np.inf

        for i in range(self.n_iter):
            q = self.e_step(X, mean, std, weights, q)
            mean, std, weights = self.m_step(X, mean, std, weights, q)
            L_next = self.likelihood(X, mean, std, weights)
            L_ratio = abs((L_next - L_prev) / L_prev)

            if L_ratio < self.tol and not np.isinf(L_prev):
                break

            L_prev = L_next

        if L_ratio >= self.tol:
            print('Convergence not attained, after {} iterations,'
                  ' relative improvement = {}'.format(self.n_iter, L_ratio))

        self.n_iter_conv = i + 1
        self.mean = mean
        self.std = std
        self.weights = weights
        self.q = q

    def predict(self, X):
        X_cond = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            X_cond[:, i] = multivariate_normal.pdf(X, mean=self.mean[i], cov=self.std[i])
        X_cond = X_cond * self.weights
        return X_cond / X_cond.sum(axis=1)[:, np.newaxis]

    def e_step(self, X, means, std, weights, q):
        for i in range(self.n_clusters):
            q[:, i] = multivariate_normal.pdf(X, mean=means[i], cov=std[i])
        q = q * weights
        return q / q.sum(axis=1)[:, np.newaxis]

    def m_step(self, X, means, std, weights, q):
        means_ = np.zeros_like(means)
        std_ = np.zeros_like(std)

        for i in range(self.n_clusters):
            means_[i] = (X * q[:, i][:, np.newaxis]).sum(axis=0) / q[:, i].sum()

        for i in range(self.n_clusters):
            centered_X = X - means_[i]
            for j in range(X.shape[0]):
                std_[i] += np.outer(centered_X[j], centered_X[j]) * q[j, i]
            std_[i] = std_[i] / q[:, i].sum()

        weights_ = q.mean(axis=0)

        return means_, std_, weights_

    def likelihood(self, X, means, std, weights):
        X_pdfs = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            X_pdfs[:, i] = multivariate_normal.pdf(X, mean=means[i], cov=std[i])

        return np.log((X_pdfs * weights).sum(axis=1)).sum()
