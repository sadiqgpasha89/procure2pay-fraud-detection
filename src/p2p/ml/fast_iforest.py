"""Low-latency Isolation Forest scoring.

scikit-learn's `score_samples` spends ~50 µs of joblib/Python overhead *per tree per call* — ~50 ms for a single row with
300 trees, which alone would blow the 850 ms budget under load. This precomputes, for every node of every tree, the value
`depth(node) + c(n_samples(node))` once, so scoring a row is one `tree_.apply` per tree plus a table lookup.
Numerically identical to `IsolationForest.score_samples` (asserted in tests)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def _c(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    return out


class FastIForest:
    def __init__(self, forest: IsolationForest):
        self.n_features = forest.n_features_in_
        self.tables: list[tuple[object, np.ndarray, np.ndarray | None]] = []
        for tree, feats in zip(forest.estimators_, forest.estimators_features_):
            t = tree.tree_
            depth = np.zeros(t.node_count)
            stack = [(0, 0)]
            while stack:
                node, d = stack.pop()
                depth[node] = d
                if t.children_left[node] != -1:
                    stack += [(t.children_left[node], d + 1), (t.children_right[node], d + 1)]
            self.tables.append((t, depth + _c(t.n_node_samples), None if len(feats) == self.n_features else feats))
        self.norm = float(_c(np.array([forest.max_samples_]))[0]) * len(self.tables)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        total = np.zeros(X32.shape[0])
        for t, table, feats in self.tables:
            leaves = t.apply(X32 if feats is None else np.ascontiguousarray(X32[:, feats]))
            total += table[leaves]
        return -(2.0 ** (-total / self.norm))
