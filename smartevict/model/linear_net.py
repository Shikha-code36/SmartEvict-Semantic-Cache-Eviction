"""Linear eviction "net": Q(x) = w . x + b, trained with the same per-candidate
MSE + Adam loop as DuelingEvictionNet (see smartevict/model/dueling_net.py),
but with no trunk, no value/advantage split, no nonlinearity -- just 7
parameters (6 weights + bias) for the same 6 input features.

Exists to isolate "does the dueling architecture's extra capacity matter"
from "does having *any* learned model beat non-learned heuristics" -- same
public interface as DuelingEvictionNet (q_values / train_batch / save /
load / n_params) so it's a drop-in swap in LearnedPolicy.
"""
from __future__ import annotations

import numpy as np


class LinearEvictionNet:
    def __init__(self, n_features: int = 6, seed: int = 0, **_ignored):
        rng = np.random.default_rng(seed)
        self.p = {
            "w": (rng.standard_normal(n_features) * 0.01).astype(np.float32),
            "b": np.zeros(1, np.float32),
        }
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.p.items()}
        self._t = 0

    def q_values(self, cand_feats: np.ndarray) -> np.ndarray:
        return cand_feats @ self.p["w"] + self.p["b"][0]

    def train_batch(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    lr: float = 1e-3) -> float:
        """groups is accepted for interface parity with DuelingEvictionNet
        (candidates sharing a decision) but unused: a plain linear model has
        no per-group context to pool, so this is ordinary per-sample MSE."""
        q = X @ self.p["w"] + self.p["b"][0]
        err = (q - y).astype(np.float32)
        loss = float(np.mean(err ** 2))
        g_q = (2.0 / len(y)) * err

        grads = {"w": X.T @ g_q, "b": np.array([g_q.sum()], np.float32)}

        self._t += 1
        b1m, b2m, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            m, s = self._adam[k]
            g = grads[k].astype(np.float32).reshape(self.p[k].shape)
            m[:] = b1m * m + (1 - b1m) * g
            s[:] = b2m * s + (1 - b2m) * g * g
            mh = m / (1 - b1m ** self._t); sh = s / (1 - b2m ** self._t)
            self.p[k] -= lr * mh / (np.sqrt(sh) + eps)
        return loss

    def save(self, path: str):
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path: str) -> "LinearEvictionNet":
        data = np.load(path)
        net = cls(n_features=data["w"].shape[0])
        for k in net.p:
            net.p[k] = data[k].astype(np.float32)
        return net

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.p.values())
