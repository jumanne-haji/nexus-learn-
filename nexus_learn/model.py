import math
import re
import numpy as np


class NumpyRegressor:
    """Small NumPy-only numerical learner.

    The representation combines character n-grams with parser-free numerical
    structure extracted from the expression.  This keeps the learner neural
    while making the signal invariant to superficial wording/formatting.
    """

    def __init__(self, seed=0, dim=256, hidden=64):
        r = np.random.default_rng(seed)
        self.dim, self.hidden = dim, hidden
        self.W1 = r.normal(0, np.sqrt(2 / dim), (dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, np.sqrt(2 / hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    @staticmethod
    def _numbers(s):
        return [float(x) for x in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", s)]

    def featurize(self, text):
        s = " ".join(text.lower().split())
        x = np.zeros(self.dim, dtype=float)
        compact = s.replace(" ", "")
        # lexical signal
        for i, ch in enumerate(compact):
            x[160 + ord(ch) % 48] += 1.0
            if i + 1 < len(compact):
                x[208 + (ord(ch) * 31 + ord(compact[i + 1])) % 48] += 0.5

        nums = self._numbers(s)
        vals = nums[:4] + [0.0] * (4 - len(nums))
        k, a, rhs, extra = vals

        # Structural type is derived from syntax, not from the answer.
        kind = 1 if "= ?" in s else (2 if "^?" in s else (3 if "log_" in s else 0))
        unknown_x = 1 if "log_" in s and "x" in s else 0

        # Scale inputs while retaining the exact algebraic quantities.
        semantic = [
            k / 10.0, a / 100.0, rhs / 10.0, extra / 1000.0,
            math.log1p(abs(k)), math.log1p(abs(a)),
            math.log1p(abs(rhs)), math.log1p(abs(extra)),
            kind, unknown_x, 1.0 if "log_" in s else 0.0,
            1.0 if "^?" in s else 0.0,
            1.0 if "= ?" in s else 0.0,
            # mathematically useful invariants
            (math.log(a) / math.log(k)) if k > 1 and a > 0 else 0.0,
            k / a if a else 0.0,
            a / k if k else 0.0,
        ]
        x[:len(semantic)] = semantic
        n = np.linalg.norm(x)
        return x / n if n else x

    def forward(self, texts):
        X = np.vstack([self.featurize(t) for t in texts])
        Z = X @ self.W1 + self.b1
        H = np.maximum(Z, 0)
        Y = H @ self.W2 + self.b2
        return X, Z, H, Y[:, 0]

    def predict(self, texts):
        z = self.forward(texts)[3]
        return np.maximum(np.expm1(z), 0.0)

    def train_batch(self, texts, y, lr=0.003):
        y = np.log1p(np.asarray(y, dtype=float))
        X, Z, H, Y = self.forward(texts)
        err = Y - y
        loss = float(np.mean(err ** 2))
        dY = (2 * err / len(y))[:, None]
        dW2 = H.T @ dY
        db2 = dY.sum(0)
        dH = dY @ self.W2.T
        dZ = dH * (Z > 0)
        dW1 = X.T @ dZ
        db1 = dZ.sum(0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
