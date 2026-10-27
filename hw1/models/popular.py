import numpy as np


class Popular:
    def fit(self, matrix):
        self.popularity = np.asarray(matrix.sum(axis=0)).ravel()
        return self

    def scores_for_user(self, user_idx):
        return self.popularity
