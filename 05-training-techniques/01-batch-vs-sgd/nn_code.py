"""
=== MODULE 05.1: BATCH vs STOCHASTIC GRADIENT DESCENT ===

So far, we've always trained using ALL the data at once:
    gradient = average gradient over ALL 200 data points
    update weights once

This is called FULL-BATCH gradient descent. It works, but has issues:
  + Smooth, stable gradient (low noise)
  - Expensive per step (must process ALL data)
  - Can get stuck in bad local minima (too smooth to escape)

The alternative is STOCHASTIC gradient descent (SGD):
    pick ONE random data point
    gradient = gradient from that single point
    update weights

  + Cheap per step (1 point vs 200)
  + Noise helps escape local minima
  - Very noisy gradient (high variance)
  - Loss curve is jagged

The sweet spot is MINI-BATCH gradient descent:
    shuffle data, split into batches of size B (e.g., 32)
    for each batch:
        gradient = average gradient over B points
        update weights

  + Moderate cost per step
  + Some noise (good for escaping minima)
  + Smoother than pure SGD
  + Can exploit GPU parallelism (batch matrix ops)
  This is what everyone actually uses!

KEY INSIGHT:
    All three methods converge to roughly the same solution.
    The difference is in HOW they get there:
    - Full batch: smooth, direct path
    - Mini-batch: slightly wiggly path
    - SGD: drunk walk that somehow gets there
"""

import numpy as np


# ============================================================
# ACTIVATIONS
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_grad_z(z):
    return (z > 0).astype(float)


# ============================================================
# SIMPLE MLP (reused from Module 04, streamlined)
# ============================================================
class MLP:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.weights.append(np.random.randn(layer_sizes[i+1], fan_in) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros(layer_sizes[i+1]))
        self.layer_sizes = layer_sizes

    def copy(self):
        """Deep copy for running parallel experiments."""
        m = MLP.__new__(MLP)
        m.layer_sizes = list(self.layer_sizes)
        m.weights = [w.copy() for w in self.weights]
        m.biases = [b.copy() for b in self.biases]
        return m

    def forward(self, X):
        self._cache_a = [X]
        self._cache_z = []
        h = X
        for i in range(len(self.weights)):
            z = h @ self.weights[i].T + self.biases[i]
            self._cache_z.append(z)
            h = sigmoid(z) if i == len(self.weights) - 1 else relu(z)
            self._cache_a.append(h)
        return h

    def loss(self, pred, labels):
        eps = 1e-7
        p = np.clip(pred, eps, 1 - eps)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def backward(self, labels):
        N = labels.shape[0]
        delta = (self._cache_a[-1] - labels) / N
        self._grad_w = []
        self._grad_b = []
        for i in range(len(self.weights) - 1, -1, -1):
            self._grad_w.insert(0, delta.T @ self._cache_a[i])
            self._grad_b.insert(0, np.sum(delta, axis=0))
            if i > 0:
                delta = (delta @ self.weights[i]) * relu_grad_z(self._cache_z[i-1])

    def update(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self._grad_w[i]
            self.biases[i] -= lr * self._grad_b[i]

    def accuracy(self, X, labels):
        pred = self.forward(X)
        correct = np.sum((pred >= 0.5).flatten() == (labels >= 0.5).flatten())
        return float(correct / len(labels))

    def predict_grid(self, rng=3.5, res=50):
        x = np.linspace(-rng, rng, res)
        X1, X2 = np.meshgrid(x, x)
        grid = np.column_stack([X1.ravel(), X2.ravel()])
        return self.forward(grid).reshape(res, res).tolist()


# ============================================================
# DATASETS
# ============================================================
def make_moons(n=200, noise=0.12, seed=42):
    np.random.seed(seed)
    n_half = n // 2
    t1 = np.linspace(0, np.pi, n_half)
    t2 = np.linspace(0, np.pi, n - n_half)
    x1 = np.column_stack([np.cos(t1), np.sin(t1)])
    x2 = np.column_stack([np.cos(t2) + 0.5, -np.sin(t2) + 0.5])
    X = np.vstack([x1, x2]) + np.random.randn(n, 2) * noise
    labels = np.array([1.0] * n_half + [0.0] * (n - n_half))
    idx = np.random.permutation(n)
    return X[idx], labels[idx].reshape(-1, 1)


# ============================================================
# TRAINING MODES
# ============================================================
def train_comparison(num_epochs=30, lr=0.5, batch_size=16, seed=42):
    """Train three identical networks with different batch strategies.

    An EPOCH = one full pass through all data.
    We track metrics per epoch so they're directly comparable.

    Returns per-epoch loss/accuracy for each mode.
    """
    X, labels = make_moons(n=200, seed=seed)
    N = len(X)

    # Three networks starting from IDENTICAL weights
    net_full = MLP([2, 16, 8, 1], seed=42)
    net_mini = net_full.copy()
    net_sgd = net_full.copy()

    results = {'full': [], 'mini': [], 'sgd': []}

    rng = np.random.RandomState(seed)

    for epoch in range(num_epochs):
        # Shuffle once per epoch (same shuffle for mini and sgd)
        idx = rng.permutation(N)
        X_shuf = X[idx]
        y_shuf = labels[idx]

        # --- FULL BATCH: one update per epoch ---
        pred = net_full.forward(X)
        net_full.backward(labels)
        net_full.update(lr)

        # --- MINI-BATCH: N/batch_size updates per epoch ---
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]
            net_mini.forward(Xb)
            net_mini.backward(yb)
            net_mini.update(lr)

        # --- SGD: N updates per epoch (batch_size=1) ---
        for i in range(N):
            Xs = X_shuf[i:i+1]
            ys = y_shuf[i:i+1]
            net_sgd.forward(Xs)
            net_sgd.backward(ys)
            net_sgd.update(lr)

        # Record epoch metrics (evaluate on full dataset)
        for name, net in [('full', net_full), ('mini', net_mini), ('sgd', net_sgd)]:
            pred = net.forward(X)
            loss = net.loss(pred, labels)
            acc = net.accuracy(X, labels)
            results[name].append({'epoch': epoch + 1, 'loss': loss, 'accuracy': acc})

    # Final heatmaps
    heatmaps = {
        'full': net_full.predict_grid(),
        'mini': net_mini.predict_grid(),
        'sgd': net_sgd.predict_grid(),
    }

    # Data points for display
    pts = [{'x1': float(X[i, 0]), 'x2': float(X[i, 1]),
             'label': 'A' if labels[i, 0] > 0.5 else 'B'} for i in range(N)]

    return {
        'results': results,
        'heatmaps': heatmaps,
        'points': pts,
        'updates_per_epoch': {
            'full': 1,
            'mini': int(np.ceil(N / batch_size)),
            'sgd': N,
        },
    }


def train_step_by_step(total_epochs=50, lr=0.5, batch_size=16, seed=42):
    """Return loss at each WEIGHT UPDATE (not just per epoch).
    Shows the noise difference between methods clearly.
    """
    X, labels = make_moons(n=200, seed=seed)
    N = len(X)

    net_full = MLP([2, 16, 8, 1], seed=42)
    net_mini = net_full.copy()
    net_sgd = net_full.copy()

    traces = {'full': [], 'mini': [], 'sgd': []}

    rng = np.random.RandomState(seed)
    update_count = {'full': 0, 'mini': 0, 'sgd': 0}

    for epoch in range(total_epochs):
        idx = rng.permutation(N)
        X_shuf = X[idx]
        y_shuf = labels[idx]

        # Full batch
        pred = net_full.forward(X)
        loss = net_full.loss(pred, labels)
        net_full.backward(labels)
        net_full.update(lr)
        update_count['full'] += 1
        traces['full'].append({'update': update_count['full'], 'loss': loss})

        # Mini-batch
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]
            pred = net_mini.forward(Xb)
            loss = net_mini.loss(pred, yb)
            net_mini.backward(yb)
            net_mini.update(lr)
            update_count['mini'] += 1
            traces['mini'].append({'update': update_count['mini'], 'loss': loss})

        # SGD
        for i in range(N):
            Xs = X_shuf[i:i+1]
            ys = y_shuf[i:i+1]
            pred = net_sgd.forward(Xs)
            loss = net_sgd.loss(pred, ys)
            net_sgd.backward(ys)
            net_sgd.update(lr)
            update_count['sgd'] += 1
            # Only record every 10th for SGD (too many points otherwise)
            if update_count['sgd'] % 10 == 0:
                full_pred = net_sgd.forward(X)
                full_loss = net_sgd.loss(full_pred, labels)
                traces['sgd'].append({'update': update_count['sgd'], 'loss': full_loss})

    return traces


if __name__ == '__main__':
    print("=" * 55)
    print("  BATCH vs STOCHASTIC vs MINI-BATCH")
    print("=" * 55)
    r = train_comparison(num_epochs=30)
    for mode in ['full', 'mini', 'sgd']:
        last = r['results'][mode][-1]
        print(f"  {mode:5s}: loss={last['loss']:.4f}  acc={last['accuracy']:.0%}  "
              f"({r['updates_per_epoch'][mode]} updates/epoch)")
