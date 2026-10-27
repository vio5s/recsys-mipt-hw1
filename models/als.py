import numpy as np
from scipy import sparse


class ALSImplicit:
    def __init__(self, n_factors=16, n_iters=4, alpha=10.0, reg=1.0, random_state=42):
        self.n_factors, self.n_iters = n_factors, n_iters
        self.alpha, self.reg, self.random_state = alpha, reg, random_state

    def fit(self, positive_matrix):
        positive_matrix = sparse.csr_matrix(positive_matrix, dtype=np.float64)
        by_item = positive_matrix.tocsc()
        n_users_fit, n_items_fit = positive_matrix.shape
        rng = np.random.default_rng(self.random_state)
        self.user_factors = np.zeros((n_users_fit, self.n_factors))
        self.item_factors = 0.05 * rng.standard_normal((n_items_fit, self.n_factors))
        eye = np.eye(self.n_factors)

        for _ in range(self.n_iters):
            item_gram = self.item_factors.T @ self.item_factors
            for u in range(n_users_fit):
                start, end = positive_matrix.indptr[u:u + 2]
                items = positive_matrix.indices[start:end]
                if len(items) == 0:
                    continue
                factors = self.item_factors[items]
                self.user_factors[u] = np.linalg.solve(
                    item_gram + self.alpha * (factors.T @ factors) + self.reg * eye,
                    (1 + self.alpha) * factors.sum(axis=0)
                )
            user_gram = self.user_factors.T @ self.user_factors
            for i in range(n_items_fit):
                start, end = by_item.indptr[i:i + 2]
                users = by_item.indices[start:end]
                if len(users) == 0:
                    self.item_factors[i] = 0
                    continue
                factors = self.user_factors[users]
                self.item_factors[i] = np.linalg.solve(
                    user_gram + self.alpha * (factors.T @ factors) + self.reg * eye,
                    (1 + self.alpha) * factors.sum(axis=0)
                )
        return self

    def scores_for_user(self, user_idx):
        return self.item_factors @ self.user_factors[user_idx]
