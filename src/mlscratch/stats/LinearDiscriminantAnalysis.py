import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None  # Store the eigenvector computed

    def fit(self, X, y):
        n_features = X.shape[1]  # 150, 4 (iris dataset)
        class_labels = np.unique(y)

        # Calculate S_W, S_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))  # 4, 4
        S_B = np.zeros((n_features, n_features))  # 4, 4

        for c in class_labels:
            # Filter X for class c
            X_c = X[y == c]

            # Calculate the mean vector for class c
            mean_c = np.mean(X_c, axis=0)

            # Calculate within-class scatter for class c
            # (4, n_c) * (n_c, 4) = (4, 4)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            # number of samples in classe C
            n_c = X_c.shape[0]

            # Difference between the overall mean and the mean of class c --> between-class scatter
            # (4, 1) * (1, 4) = (4, 4)
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)  # (4, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(S_W).dot(S_B)

        # Get the eigenvalues and eigenvectors of (SW^-1 * SB)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Keep only the real parts of eigenvalues and eigenvectors
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors.T)

        # Sort the eigenvalues descending (high to low)
        idxs = np.argsort(np.abs(eigenvalues))[::-1]
        self.eigenvalues = np.abs(eigenvalues[idxs])
        self.eigenvectors = eigenvectors[idxs]

        # Store the first m eigenvectors as transformation matrix W
        self.W = self.eigenvectors[0 : self.n_components]

    def transform(self, X):
        return np.dot(
            X, self.W.T
        )  # Project the data into the linear discriminant vector (w) calculated

    def _get_explained_variance(self):
        return self.eigenvalues[0 : self.n_components] / np.sum(self.eigenvalues)
