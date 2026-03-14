"""
=== MODULE 05.3: OVERFITTING & REGULARIZATION ===

THE PROBLEM: OVERFITTING

  A network with enough neurons can memorize ANY dataset perfectly.
  But memorizing ≠ understanding. If the data has noise (and real data
  always does), the network learns the noise too.

  Signs of overfitting:
    - Training loss keeps going down
    - Test loss starts going UP (the network gets worse on new data)
    - Decision boundary becomes wildly complex, tracing every noise point

  This is the same as curve fitting with too many basis functions:
  a 100th-degree polynomial can pass through every point, but it
  oscillates wildly between them. It "fits" perfectly but "generalizes"
  terribly. Same principle here.

SOLUTION 1: L2 REGULARIZATION (Weight Decay)

  Add a penalty for large weights to the loss:

    loss_total = loss_data + λ * Σ(w²)

  This discourages the network from using large weights.
  Large weights → sharp, complex decision boundaries.
  Small weights → smooth, simple decision boundaries.
  λ controls the tradeoff: bigger λ = simpler model.

  Implementation: during the weight update, shrink each weight slightly:
    w -= lr * (gradient + λ * w)

  This is called "weight decay" because weights literally decay toward zero.

SOLUTION 2: DROPOUT

  During training, randomly set some neurons to zero (typically 50%).
  Each batch sees a different random subset of neurons.

  Why this works:
    - Forces redundancy — no single neuron can memorize a pattern
    - Like training an ensemble of smaller networks
    - Each neuron must learn features that are useful INDEPENDENTLY

  During testing, use ALL neurons but scale outputs by the dropout rate.
  (Or equivalently, scale up during training — "inverted dropout")

HOW TO DETECT OVERFITTING:
  Split your data into TRAINING set (what the network learns from)
  and TEST set (what you evaluate on, never trained on).
  If training loss drops but test loss rises → overfitting.
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
# MLP WITH REGULARIZATION OPTIONS
# ============================================================
class MLP:
    def __init__(self, layer_sizes=[2, 32, 16, 1], seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.weights.append(np.random.randn(layer_sizes[i+1], fan_in) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros(layer_sizes[i+1]))
        self.train_history = []
        self.rng = np.random.RandomState(seed + 1)

    def copy(self):
        m = MLP.__new__(MLP)
        m.layer_sizes = list(self.layer_sizes)
        m.weights = [w.copy() for w in self.weights]
        m.biases = [b.copy() for b in self.biases]
        m.train_history = []
        m.rng = np.random.RandomState(self.rng.randint(100000))
        return m

    def forward(self, X, dropout_rate=0.0, training=False):
        self._a = [X]; self._z = []; self._masks = []
        h = X
        for i in range(len(self.weights)):
            z = h @ self.weights[i].T + self.biases[i]
            self._z.append(z)
            if i == len(self.weights) - 1:
                h = sigmoid(z)
            else:
                h = relu(z)
                # Apply dropout to hidden layers during training
                if training and dropout_rate > 0:
                    mask = (self.rng.rand(*h.shape) > dropout_rate).astype(float)
                    h = h * mask / (1 - dropout_rate)  # inverted dropout
                    self._masks.append(mask)
                else:
                    self._masks.append(np.ones_like(h))
            self._a.append(h)
        return h

    def loss(self, pred, labels):
        eps = 1e-7; p = np.clip(pred, eps, 1 - eps)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def l2_penalty(self):
        return sum(float(np.sum(w**2)) for w in self.weights)

    def backward(self, labels, dropout_rate=0.0):
        N = labels.shape[0]
        delta = (self._a[-1] - labels) / N
        self._gw = []; self._gb = []
        for i in range(len(self.weights) - 1, -1, -1):
            self._gw.insert(0, delta.T @ self._a[i])
            self._gb.insert(0, np.sum(delta, axis=0))
            if i > 0:
                delta = delta @ self.weights[i]
                delta = delta * relu_grad_z(self._z[i-1])
                # Apply same dropout mask
                if dropout_rate > 0 and i - 1 < len(self._masks):
                    delta = delta * self._masks[i-1] / (1 - dropout_rate)

    def update(self, lr, l2_lambda=0.0):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * (self._gw[i] + l2_lambda * self.weights[i])
            self.biases[i] -= lr * self._gb[i]

    def accuracy(self, X, labels):
        pred = self.forward(X, training=False)
        return float(np.sum((pred >= 0.5).flatten() == (labels >= 0.5).flatten()) / len(labels))

    def predict_grid(self, rng=2.5, res=50):
        x = np.linspace(-rng, rng, res)
        X1, X2 = np.meshgrid(x, x)
        grid = np.column_stack([X1.ravel(), X2.ravel()])
        return self.forward(grid, training=False).reshape(res, res).tolist()


# ============================================================
# DATASET WITH TRAIN/TEST SPLIT
# ============================================================
def make_noisy_moons(n_train=80, n_test=200, noise=0.25, seed=42):
    """Small noisy training set (easy to overfit) + large clean test set."""
    def _moons(n, noise, seed):
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

    X_train, y_train = _moons(n_train, noise, seed)
    X_test, y_test = _moons(n_test, noise * 0.5, seed + 100)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'train_points': [{'x1': float(X_train[i, 0]), 'x2': float(X_train[i, 1]),
                          'label': 'A' if y_train[i, 0] > 0.5 else 'B'} for i in range(n_train)],
        'test_points': [{'x1': float(X_test[i, 0]), 'x2': float(X_test[i, 1]),
                         'label': 'A' if y_test[i, 0] > 0.5 else 'B'} for i in range(n_test)],
    }


# ============================================================
# COMPARISON: None vs L2 vs Dropout
# ============================================================
def train_comparison(epochs=200, lr=0.5, l2_lambda=0.01, dropout_rate=0.5, seed=42):
    """Train three identical networks: no reg, L2, dropout."""
    ds = make_noisy_moons(n_train=80, n_test=200, noise=0.25, seed=seed)
    X_tr, y_tr = ds['X_train'], ds['y_train']
    X_te, y_te = ds['X_test'], ds['y_test']

    base = MLP([2, 32, 16, 1], seed=42)
    nets = {
        'none': base.copy(),
        'l2': base.copy(),
        'dropout': base.copy(),
    }

    results = {name: [] for name in nets}

    for epoch in range(epochs):
        for name, net in nets.items():
            do_rate = dropout_rate if name == 'dropout' else 0.0
            l2 = l2_lambda if name == 'l2' else 0.0

            # Train
            pred = net.forward(X_tr, dropout_rate=do_rate, training=True)
            train_loss = net.loss(pred, y_tr)
            if name == 'l2':
                train_loss += l2 * net.l2_penalty()
            net.backward(y_tr, dropout_rate=do_rate)
            net.update(lr, l2_lambda=l2)

            # Evaluate (no dropout)
            train_acc = net.accuracy(X_tr, y_tr)
            pred_te = net.forward(X_te, training=False)
            test_loss = net.loss(pred_te, y_te)
            test_acc = net.accuracy(X_te, y_te)

            results[name].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
            })

    heatmaps = {name: net.predict_grid() for name, net in nets.items()}

    return {
        'results': results,
        'heatmaps': heatmaps,
        'train_points': ds['train_points'],
        'test_points': ds['test_points'],
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  OVERFITTING & REGULARIZATION")
    print("=" * 55)
    r = train_comparison(epochs=200)
    for name in ['none', 'l2', 'dropout']:
        last = r['results'][name][-1]
        print(f"  {name:8s}: train_acc={last['train_acc']:.0%}  test_acc={last['test_acc']:.0%}  "
              f"train_loss={last['train_loss']:.4f}  test_loss={last['test_loss']:.4f}")
