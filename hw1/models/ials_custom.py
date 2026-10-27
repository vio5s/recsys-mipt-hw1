import numpy as np
from scipy import sparse


def custom_target(frame, end_ms, alpha=10.0, show_weight=0.3, half_life_days=7.0):
    decay = 2.0 ** (-(end_ms - frame.time_ms) / 86_400_000 / half_life_days)
    engagement = 1 + frame.long_view + frame.is_like + frame.is_follow + frame.is_comment + frame.is_forward
    signal = np.where(frame.is_click == 1, engagement, show_weight)
    signal = np.where(frame.is_hate == 1, -1.0, signal) * decay
    return np.sign(signal) * (1 + alpha * np.abs(signal))


def update_rows(C, Y, reg, batch_size=32):
    C = C.tocsr()
    d = Y.shape[1]
    base = Y.T @ Y + reg * np.eye(d)
    result = np.empty((C.shape[0], d), dtype=np.float64)
    for begin in range(0, C.shape[0], batch_size):
        end = min(begin + batch_size, C.shape[0])
        A = np.broadcast_to(base, (end - begin, d, d)).copy()
        b = np.zeros((end - begin, d))
        for offset, u in enumerate(range(begin, end)):
            start, stop = C.indptr[u:u + 2]
            idx, value = C.indices[start:stop], C.data[start:stop]
            local = Y[idx]
            c = np.abs(value)
            A[offset] += local.T @ ((c - 1)[:, None] * local)
            b[offset] = local.T @ np.where(value > 0, c, 0.)
        result[begin:end] = np.linalg.solve(A, b[..., None])[..., 0]
    return result


class CustomTargetALS:
    def __init__(self, n_factors=32, n_iters=10, reg=1.0, random_state=42):
        self.n_factors, self.n_iters = n_factors, n_iters
        self.reg, self.random_state = reg, random_state

    def fit(self, C):
        C = sparse.csr_matrix(C, dtype=np.float64)
        C.eliminate_zeros()
        C_items = C.T.tocsr()
        rng = np.random.default_rng(self.random_state)
        self.item_factors = 0.05 * rng.standard_normal((C.shape[1], self.n_factors))
        for _ in range(self.n_iters):
            self.user_factors = update_rows(C, self.item_factors, self.reg)
            self.item_factors = update_rows(C_items, self.user_factors, self.reg)
        return self

    def scores_for_user(self, user_idx):
        return self.item_factors @ self.user_factors[user_idx]
