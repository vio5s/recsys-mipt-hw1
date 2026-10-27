import numpy as np
from scipy import sparse

from models.utils import user_items


class ItemKNN:
    def __init__(self, k=100, shrinkage=10):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, matrix):
        self.matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        overlap = (self.matrix.T @ self.matrix).toarray()
        norms = np.sqrt(np.diag(overlap))
        denominator = np.outer(norms, norms)
        similarity = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
        similarity *= np.divide(overlap, overlap + self.shrinkage, out=np.zeros_like(overlap), where=overlap > 0)
        np.fill_diagonal(similarity, 0)
        weak = np.argpartition(-similarity, self.k, axis=1)[:, self.k:]
        np.put_along_axis(similarity, weak, 0, axis=1)
        self.similarity = similarity
        return self

    def scores_for_user(self, user_idx):
        return self.similarity[user_items(self.matrix, user_idx)].sum(axis=0)
