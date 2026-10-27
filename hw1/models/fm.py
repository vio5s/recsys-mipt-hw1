import numpy as np
from scipy import sparse


def sample_negatives(rng, user_indices, positive_matrix):
    negative = rng.integers(0, positive_matrix.shape[1], size=len(user_indices))
    collisions = np.asarray(positive_matrix[user_indices, negative]).ravel() > 0
    while np.any(collisions):
        negative[collisions] = rng.integers(0, positive_matrix.shape[1], size=int(collisions.sum()))
        collisions = np.asarray(positive_matrix[user_indices, negative]).ravel() > 0
    return negative


class FactorizationMachine:
    def __init__(self, n_factors=16, n_epochs=20, lr=0.01, reg_w=0.01, reg_v=0.01,
                 batch_size=512, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.batch_size = batch_size
        self.random_state = random_state

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.w0_ = 0.0
        self.w_ = np.zeros(n_features, dtype=np.float32)
        self.V_ = (0.01 * rng.standard_normal((self.n_factors, n_features))).astype(np.float32)

    def _forward(self, X):
        X_sq = X.copy()
        X_sq.data **= 2
        linear = np.asarray(X @ self.w_).ravel()
        Vx = np.asarray(X @ self.V_.T, dtype=np.float32)
        V2x2 = np.asarray(X_sq @ (self.V_ ** 2).T, dtype=np.float32)
        interaction = 0.5 * np.sum(Vx ** 2 - V2x2, axis=1)
        return self.w0_ + linear + interaction, Vx, X_sq

    def _step_bpr(self, X_pos, X_neg):
        y_pos, Vx_pos, Xsq_pos = self._forward(X_pos)
        y_neg, Vx_neg, Xsq_neg = self._forward(X_neg)

        sigma = 1.0 / (1.0 + np.exp(-np.clip(y_pos - y_neg, -30, 30)))
        e_pos = (sigma - 1.0).astype(np.float32)
        e_neg = (1.0 - sigma).astype(np.float32)
        counts = np.maximum(X_pos.getnnz(axis=0) + X_neg.getnnz(axis=0), 1).astype(np.float32)

        grad_w = (np.asarray(X_pos.T @ e_pos).ravel() + np.asarray(X_neg.T @ e_neg).ravel()) / counts
        self.w_ -= self.lr * (grad_w + 2 * self.reg_w * self.w_)

        term1 = X_pos.T @ (e_pos[:, None] * Vx_pos) + X_neg.T @ (e_neg[:, None] * Vx_neg)
        term2 = np.asarray(Xsq_pos.T @ e_pos).ravel() + np.asarray(Xsq_neg.T @ e_neg).ravel()
        grad_VT = (np.asarray(term1, dtype=np.float32) - term2[:, None] * self.V_.T) / counts[:, None]
        self.V_ -= self.lr * (grad_VT.T + 2 * self.reg_v * self.V_)

    def _build_X(self, user_indices, item_indices):
        return self._X_users[user_indices] + self._X_items[item_indices]

    def fit(self, positive_matrix, user_features, item_features):
        positive_matrix = sparse.csr_matrix(positive_matrix)
        n_users, n_items = positive_matrix.shape
        d_user, d_item = user_features.shape[1], item_features.shape[1]
        self._X_users = sparse.hstack(
            [sparse.identity(n_users), user_features, sparse.csr_matrix((n_users, n_items + d_item))],
            format="csr", dtype=np.float32,
        )
        self._X_items = sparse.hstack(
            [sparse.csr_matrix((n_items, n_users + d_user)), sparse.identity(n_items), item_features],
            format="csr", dtype=np.float32,
        )
        self._init_params(self._X_users.shape[1])

        user_indices, item_indices = positive_matrix.nonzero()
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_epochs):
            order = rng.permutation(len(user_indices))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                u_batch = user_indices[idx]
                j_batch = sample_negatives(rng, u_batch, positive_matrix)
                self._step_bpr(self._build_X(u_batch, item_indices[idx]), self._build_X(u_batch, j_batch))

        self.item_bias_, self.item_emb_, _ = self._forward(self._X_items)
        self.user_emb_ = np.asarray(self._X_users @ self.V_.T)
        return self

    def scores_for_user(self, user_idx):
        return self.item_bias_ + self.item_emb_ @ self.user_emb_[user_idx]
