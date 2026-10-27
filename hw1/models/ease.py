import numpy as np
from scipy import linalg, sparse

from models.utils import user_items


class EASE:
    def __init__(self, reg=100.0):
        self.reg = reg

    def fit(self, matrix):
        self.history = sparse.csr_matrix(matrix, dtype=np.float64).copy()
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        factor = linalg.cho_factor(gram, lower=True, check_finite=False)
        precision = linalg.cho_solve(factor, np.eye(gram.shape[0]), check_finite=False)
        self.W = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.W, 0)
        return self

    def scores_for_user(self, user_idx):
        return self.W[user_items(self.history, user_idx)].sum(axis=0)
