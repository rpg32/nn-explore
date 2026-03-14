"""
=== MODULE 05.4: DATA NORMALIZATION ===

THE PROBLEM:

  Imagine two input features:
    x1 = house size in square feet (500 to 5000)
    x2 = number of bedrooms (1 to 6)

  A neuron computes: w1*x1 + w2*x2 + b

  The gradient for w1 is proportional to x1 (which is huge: ~2000)
  The gradient for w2 is proportional to x2 (which is tiny: ~3)

  So w1 gets massive gradients and w2 gets tiny ones.
  Gradient descent oscillates wildly in the w1 direction
  and barely moves in the w2 direction.

  This is the elongated bowl problem from Module 05.2 —
  but caused by the DATA, not the landscape shape.

THE FIX: NORMALIZE YOUR INPUTS

  Method 1: STANDARDIZATION (z-score)
    x_norm = (x - mean) / std
    Each feature ends up with mean=0, std=1

  Method 2: MIN-MAX SCALING
    x_norm = (x - min) / (max - min)
    Each feature ends up in range [0, 1]

  Standardization is more common in practice because it
  handles outliers better and doesn't bound the range.

WHY IT WORKS:
  After normalization, all features live in similar ranges.
  Gradients for all weights are similar magnitude.
  The loss landscape becomes more circular (less elongated).
  Gradient descent converges much faster.

BATCH NORMALIZATION (BatchNorm):
  Normalizes the ACTIVATIONS between layers, not just the input.
  Each hidden layer's outputs get normalized to mean=0, std=1.
  This keeps the internal signals well-behaved as the network gets deeper.
  Used in almost every modern deep network.
"""

import numpy as np


# ============================================================
# ACTIVATIONS & MLP
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_grad_z(z):
    return (z > 0).astype(float)


class MLP:
    def __init__(self, layer_sizes=[2, 16, 8, 1], seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.weights.append(np.random.randn(layer_sizes[i+1], fan_in) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros(layer_sizes[i+1]))

    def copy(self):
        m = MLP.__new__(MLP)
        m.weights = [w.copy() for w in self.weights]
        m.biases = [b.copy() for b in self.biases]
        return m

    def forward(self, X):
        self._a = [X]; self._z = []
        h = X
        for i in range(len(self.weights)):
            z = h @ self.weights[i].T + self.biases[i]
            self._z.append(z)
            h = sigmoid(z) if i == len(self.weights) - 1 else relu(z)
            self._a.append(h)
        return h

    def backward(self, labels):
        N = labels.shape[0]
        delta = (self._a[-1] - labels) / N
        self._gw = []; self._gb = []
        for i in range(len(self.weights) - 1, -1, -1):
            self._gw.insert(0, delta.T @ self._a[i])
            self._gb.insert(0, np.sum(delta, axis=0))
            if i > 0:
                delta = (delta @ self.weights[i]) * relu_grad_z(self._z[i-1])

    def update(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self._gw[i]
            self.biases[i] -= lr * self._gb[i]

    def loss(self, pred, labels):
        eps = 1e-7; p = np.clip(pred, eps, 1 - eps)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def accuracy(self, X, labels):
        pred = self.forward(X)
        return float(np.sum((pred >= 0.5).flatten() == (labels >= 0.5).flatten()) / len(labels))

    def predict_grid(self, x1_range, x2_range, res=50):
        x1 = np.linspace(x1_range[0], x1_range[1], res)
        x2 = np.linspace(x2_range[0], x2_range[1], res)
        X1, X2 = np.meshgrid(x1, x2)
        grid = np.column_stack([X1.ravel(), X2.ravel()])
        return self.forward(grid).reshape(res, res).tolist()

    def gradient_magnitudes(self):
        """Return per-weight gradient magnitudes for visualization."""
        if not hasattr(self, '_gw'):
            return []
        return [float(np.mean(np.abs(gw))) for gw in self._gw]


# ============================================================
# DATASET: Deliberately mis-scaled features
# ============================================================
def make_scaled_dataset(n=200, scale_ratio=100, seed=42):
    """Two-moon dataset where x1 is scaled up enormously.

    Without normalization: x1 ranges [~-150, 150], x2 ranges [~-1.5, 1.5]
    The network struggles because gradients for w1 are 100x larger than w2.
    """
    np.random.seed(seed)
    n_half = n // 2
    t1 = np.linspace(0, np.pi, n_half)
    t2 = np.linspace(0, np.pi, n - n_half)
    x1 = np.column_stack([np.cos(t1), np.sin(t1)])
    x2 = np.column_stack([np.cos(t2) + 0.5, -np.sin(t2) + 0.5])
    X = np.vstack([x1, x2]) + np.random.randn(n, 2) * 0.12
    labels = np.array([1.0] * n_half + [0.0] * (n - n_half))
    idx = np.random.permutation(n)
    X = X[idx]
    labels = labels[idx].reshape(-1, 1)

    # Scale x1 by scale_ratio — simulates "house size" vs "bedrooms"
    X_raw = X.copy()
    X_raw[:, 0] *= scale_ratio

    return X_raw, labels


def normalize_standardize(X):
    """Z-score: (x - mean) / std"""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def normalize_minmax(X):
    """Min-max: (x - min) / (max - min)"""
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    rng = mx - mn + 1e-8
    return (X - mn) / rng, mn, rng


def train_comparison(epochs=100, lr=0.5, scale_ratio=100, seed=42):
    """Train three identical networks: raw, standardized, min-max."""
    X_raw, labels = make_scaled_dataset(n=200, scale_ratio=scale_ratio, seed=seed)
    X_std, std_mean, std_std = normalize_standardize(X_raw)
    X_mm, mm_min, mm_rng = normalize_minmax(X_raw)

    base = MLP([2, 16, 8, 1], seed=42)

    configs = {
        'raw': {'X': X_raw, 'net': base.copy(), 'lr': lr * 0.0001},  # needs tiny LR
        'standardized': {'X': X_std, 'net': base.copy(), 'lr': lr},
        'minmax': {'X': X_mm, 'net': base.copy(), 'lr': lr},
    }

    results = {name: [] for name in configs}
    grad_history = {name: [] for name in configs}

    for epoch in range(epochs):
        for name, cfg in configs.items():
            pred = cfg['net'].forward(cfg['X'])
            loss = cfg['net'].loss(pred, labels)
            acc = cfg['net'].accuracy(cfg['X'], labels)
            cfg['net'].backward(labels)
            grad_mags = cfg['net'].gradient_magnitudes()
            cfg['net'].update(cfg['lr'])
            results[name].append({'epoch': epoch + 1, 'loss': loss, 'accuracy': acc})
            grad_history[name].append(grad_mags)

    # Data range info
    data_info = {
        'raw': {
            'x1_range': [float(X_raw[:, 0].min()), float(X_raw[:, 0].max())],
            'x2_range': [float(X_raw[:, 1].min()), float(X_raw[:, 1].max())],
            'x1_mean': float(X_raw[:, 0].mean()), 'x1_std': float(X_raw[:, 0].std()),
            'x2_mean': float(X_raw[:, 1].mean()), 'x2_std': float(X_raw[:, 1].std()),
        },
        'standardized': {
            'x1_range': [float(X_std[:, 0].min()), float(X_std[:, 0].max())],
            'x2_range': [float(X_std[:, 1].min()), float(X_std[:, 1].max())],
        },
        'minmax': {
            'x1_range': [float(X_mm[:, 0].min()), float(X_mm[:, 0].max())],
            'x2_range': [float(X_mm[:, 1].min()), float(X_mm[:, 1].max())],
        },
    }

    # Points for display
    def make_pts(X):
        return [{'x1': float(X[i, 0]), 'x2': float(X[i, 1]),
                 'label': 'A' if labels[i, 0] > 0.5 else 'B'} for i in range(len(X))]

    # Heatmaps
    heatmaps = {}
    for name, cfg in configs.items():
        x1r = data_info[name]['x1_range']
        x2r = data_info[name]['x2_range']
        pad1 = (x1r[1] - x1r[0]) * 0.1
        pad2 = (x2r[1] - x2r[0]) * 0.1
        heatmaps[name] = cfg['net'].predict_grid(
            [x1r[0] - pad1, x1r[1] + pad1],
            [x2r[0] - pad2, x2r[1] + pad2],
        )

    return {
        'results': results,
        'grad_history': grad_history,
        'heatmaps': heatmaps,
        'data_info': data_info,
        'points': {name: make_pts(cfg['X']) for name, cfg in configs.items()},
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  DATA NORMALIZATION")
    print("=" * 55)

    X_raw, labels = make_scaled_dataset(scale_ratio=100)
    print(f"  Raw data:    x1 range [{X_raw[:, 0].min():.1f}, {X_raw[:, 0].max():.1f}]"
          f"  x2 range [{X_raw[:, 1].min():.2f}, {X_raw[:, 1].max():.2f}]")

    X_std, _, _ = normalize_standardize(X_raw)
    print(f"  Standardized: x1 range [{X_std[:, 0].min():.2f}, {X_std[:, 0].max():.2f}]"
          f"  x2 range [{X_std[:, 1].min():.2f}, {X_std[:, 1].max():.2f}]")

    r = train_comparison(epochs=100)
    for name in ['raw', 'standardized', 'minmax']:
        last = r['results'][name][-1]
        print(f"  {name:14s}: loss={last['loss']:.4f}  acc={last['accuracy']:.0%}")
