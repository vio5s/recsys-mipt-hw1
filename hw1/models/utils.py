import numpy as np
from scipy import sparse


def to_csr(frame, n_users, n_items, values=None):
    if values is None:
        values = np.ones(len(frame))
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float64), (frame.user_idx.to_numpy(), frame.item_idx.to_numpy())),
        shape=(n_users, n_items),
    )


def user_items(matrix, user_idx):
    return matrix.indices[matrix.indptr[user_idx]:matrix.indptr[user_idx + 1]]


def top_indices(scores, seen_indices, candidate_mask, k=10):
    masked = np.asarray(scores, dtype=float).copy()
    masked[~candidate_mask] = -np.inf
    masked[seen_indices] = -np.inf
    candidates = np.flatnonzero(np.isfinite(masked))
    k = min(k, len(candidates))
    top = candidates[np.argpartition(-masked[candidates], k - 1)[:k]]
    return top[np.argsort(-masked[top], kind="stable")]


def evaluate(model, seen_matrix, heldout, candidate_mask, k=10):
    ndcg, recall = [], []
    recommended = set()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for user_idx, true_items in heldout.items():
        seen = user_items(seen_matrix, user_idx)
        top = top_indices(model.scores_for_user(user_idx), seen, candidate_mask, k)
        recommended.update(top.tolist())
        hits = np.isin(top, true_items)
        ndcg.append(np.sum(hits * discounts[:len(top)]) / np.sum(discounts[:min(k, len(true_items))]))
        recall.append(hits.sum() / len(true_items))
    return {
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"recall@{k}": float(np.mean(recall)),
        "coverage": len(recommended) / candidate_mask.sum(),
    }
