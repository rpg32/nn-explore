"""
=== MODULE 05.2: OPTIMIZERS ===

Plain gradient descent has a problem: it treats every direction equally.
The learning rate is the same for steep gradients and shallow gradients.

PICTURE A LONG NARROW VALLEY:
  - The gradient along the valley floor is small (shallow slope)
  - The gradient across the valley walls is large (steep sides)
  - Plain SGD oscillates across the valley while barely moving along it
  - It's like a ball bouncing side-to-side instead of rolling downhill

MOMENTUM fixes this by adding "velocity":
  v = β * v_prev + gradient          (accumulate past gradients)
  weights -= lr * v

  Past gradients along the valley floor ADD UP → faster movement
  Past gradients across the valley CANCEL OUT → less oscillation
  β is typically 0.9 (90% of previous velocity carries forward)

  Analogy: a heavy ball rolling downhill. It builds speed in consistent
  directions and dampens out jitter.

RMSPROP adapts the learning rate per-parameter:
  s = β * s_prev + (1-β) * gradient²   (track gradient magnitude)
  weights -= lr * gradient / √(s + ε)

  Parameters with large gradients get SMALLER effective learning rate
  Parameters with small gradients get LARGER effective learning rate
  This automatically handles the valley problem — steep directions slow down

ADAM combines both ideas:
  m = β1 * m_prev + (1-β1) * gradient           (momentum)
  v = β2 * v_prev + (1-β2) * gradient²          (RMSProp)
  weights -= lr * m_corrected / √(v_corrected + ε)

  m keeps the momentum (consistent direction wins)
  v adapts the learning rate (steep directions slow down)
  "corrected" = bias correction for early steps

  Adam is the DEFAULT optimizer in modern deep learning.
  It works well out of the box with lr=0.001 for most problems.
"""

import numpy as np


# ============================================================
# 2D OPTIMIZATION LANDSCAPES
# ============================================================
def rosenbrock(x, y, a=1, b=100):
    """Classic banana-shaped valley. Minimum at (a, a²) = (1, 1).
    Famous for being hard to optimize — long curved narrow valley."""
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    dx = -2*(a - x) + b * 2*(y - x**2) * (-2*x)
    dy = b * 2*(y - x**2)
    return dx, dy

def elongated_bowl(x, y):
    """Axis-aligned narrow valley. Minimum at (0, 0).
    Simple case where momentum helps most."""
    return 0.5 * x**2 + 20 * y**2

def elongated_bowl_grad(x, y):
    return x, 40 * y

def saddle(x, y):
    """Saddle point at origin — flat in one direction, curved in others."""
    return x**2 - y**2

def saddle_grad(x, y):
    return 2*x, -2*y

LANDSCAPES = {
    'bowl': {
        'name': 'Elongated Bowl',
        'fn': elongated_bowl, 'grad': elongated_bowl_grad,
        'start': (-3.0, -0.5), 'range': 4.0,
        'desc': 'Steep across, shallow along — momentum stops the oscillation',
    },
    'rosenbrock': {
        'name': 'Rosenbrock Valley',
        'fn': rosenbrock, 'grad': rosenbrock_grad,
        'start': (-1.5, 2.0), 'range': 3.0,
        'desc': 'Curved narrow valley — tests whether optimizer can follow the bend',
    },
}


# ============================================================
# OPTIMIZERS
# ============================================================
class SGDOptimizer:
    def __init__(self):
        self.name = 'SGD'

    def step(self, x, y, grad_fn, lr):
        dx, dy = grad_fn(x, y)
        return x - lr * dx, y - lr * dy

class MomentumOptimizer:
    def __init__(self, beta=0.9):
        self.name = 'Momentum'
        self.beta = beta
        self.vx = 0.0
        self.vy = 0.0

    def step(self, x, y, grad_fn, lr):
        dx, dy = grad_fn(x, y)
        self.vx = self.beta * self.vx + dx
        self.vy = self.beta * self.vy + dy
        return x - lr * self.vx, y - lr * self.vy

class RMSPropOptimizer:
    def __init__(self, beta=0.999, eps=1e-8):
        self.name = 'RMSProp'
        self.beta = beta
        self.eps = eps
        self.sx = 0.0
        self.sy = 0.0

    def step(self, x, y, grad_fn, lr):
        dx, dy = grad_fn(x, y)
        self.sx = self.beta * self.sx + (1 - self.beta) * dx**2
        self.sy = self.beta * self.sy + (1 - self.beta) * dy**2
        return (x - lr * dx / (np.sqrt(self.sx) + self.eps),
                y - lr * dy / (np.sqrt(self.sy) + self.eps))

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.name = 'Adam'
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mx = 0.0; self.my = 0.0
        self.vx = 0.0; self.vy = 0.0
        self.t = 0

    def step(self, x, y, grad_fn, lr):
        dx, dy = grad_fn(x, y)
        self.t += 1
        self.mx = self.beta1 * self.mx + (1 - self.beta1) * dx
        self.my = self.beta1 * self.my + (1 - self.beta1) * dy
        self.vx = self.beta2 * self.vx + (1 - self.beta2) * dx**2
        self.vy = self.beta2 * self.vy + (1 - self.beta2) * dy**2
        # Bias correction
        mx_hat = self.mx / (1 - self.beta1**self.t)
        my_hat = self.my / (1 - self.beta1**self.t)
        vx_hat = self.vx / (1 - self.beta2**self.t)
        vy_hat = self.vy / (1 - self.beta2**self.t)
        return (x - lr * mx_hat / (np.sqrt(vx_hat) + self.eps),
                y - lr * my_hat / (np.sqrt(vy_hat) + self.eps))


def run_optimizers(landscape='bowl', lr=0.01, steps=150):
    """Run all four optimizers on the same landscape."""
    ls = LANDSCAPES[landscape]
    grad_fn = ls['grad']
    x0, y0 = ls['start']

    optimizers = [
        SGDOptimizer(),
        MomentumOptimizer(beta=0.9),
        RMSPropOptimizer(beta=0.999),
        AdamOptimizer(beta1=0.9, beta2=0.999),
    ]

    # Adjust LR per optimizer for fair comparison
    lr_mult = {
        'SGD': 1.0,
        'Momentum': 1.0,
        'RMSProp': 10.0,     # RMSProp/Adam typically use larger lr
        'Adam': 10.0,
    }

    paths = {}
    for opt in optimizers:
        x, y = x0, y0
        path = [{'x': x, 'y': y, 'loss': float(ls['fn'](x, y))}]
        effective_lr = lr * lr_mult[opt.name]
        for _ in range(steps):
            x, y = opt.step(x, y, grad_fn, effective_lr)
            # Clamp to prevent explosion
            rng = ls['range'] * 1.5
            x = max(-rng, min(rng, x))
            y = max(-rng, min(rng, y))
            path.append({'x': float(x), 'y': float(y), 'loss': float(ls['fn'](x, y))})
        paths[opt.name] = path

    # Compute contour data
    rng = ls['range']
    res = 80
    xs = np.linspace(-rng, rng, res)
    ys = np.linspace(-rng, rng, res)
    X, Y = np.meshgrid(xs, ys)
    Z = ls['fn'](X, Y)
    # Log scale for better contour visibility
    Z_log = np.log(Z + 1e-6)

    return {
        'paths': paths,
        'contour': Z_log.tolist(),
        'range': rng,
        'landscape': ls['name'],
        'desc': ls['desc'],
    }


# ============================================================
# NEURAL NETWORK COMPARISON
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_grad_z(z):
    return (z > 0).astype(float)


class MLP:
    def __init__(self, seed=42):
        np.random.seed(seed)
        sizes = [2, 16, 8, 1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            self.weights.append(np.random.randn(sizes[i+1], fan_in) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros(sizes[i+1]))

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

    def loss(self, pred, labels):
        eps = 1e-7; p = np.clip(pred, eps, 1 - eps)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def accuracy(self, X, labels):
        pred = self.forward(X)
        return float(np.sum((pred >= 0.5).flatten() == (labels >= 0.5).flatten()) / len(labels))

    def predict_grid(self, rng=3.5, res=50):
        x = np.linspace(-rng, rng, res)
        X1, X2 = np.meshgrid(x, x)
        return self.forward(np.column_stack([X1.ravel(), X2.ravel()])).reshape(res, res).tolist()


class MLPWithOptimizer:
    """Wraps MLP with a specific optimizer for weight updates."""

    def __init__(self, net, optimizer_name='sgd'):
        self.net = net
        self.name = optimizer_name
        n_params = sum(w.size + b.size for w, b in zip(net.weights, net.biases))

        if optimizer_name == 'sgd':
            pass
        elif optimizer_name == 'momentum':
            self.vw = [np.zeros_like(w) for w in net.weights]
            self.vb = [np.zeros_like(b) for b in net.biases]
            self.beta = 0.9
        elif optimizer_name == 'rmsprop':
            self.sw = [np.zeros_like(w) for w in net.weights]
            self.sb = [np.zeros_like(b) for b in net.biases]
            self.beta = 0.999
            self.eps = 1e-8
        elif optimizer_name == 'adam':
            self.mw = [np.zeros_like(w) for w in net.weights]
            self.mb = [np.zeros_like(b) for b in net.biases]
            self.vw = [np.zeros_like(w) for w in net.weights]
            self.vb = [np.zeros_like(b) for b in net.biases]
            self.beta1 = 0.9; self.beta2 = 0.999; self.eps = 1e-8
            self.t = 0

    def update(self, lr):
        net = self.net
        if self.name == 'sgd':
            for i in range(len(net.weights)):
                net.weights[i] -= lr * net._gw[i]
                net.biases[i] -= lr * net._gb[i]

        elif self.name == 'momentum':
            for i in range(len(net.weights)):
                self.vw[i] = self.beta * self.vw[i] + net._gw[i]
                self.vb[i] = self.beta * self.vb[i] + net._gb[i]
                net.weights[i] -= lr * self.vw[i]
                net.biases[i] -= lr * self.vb[i]

        elif self.name == 'rmsprop':
            for i in range(len(net.weights)):
                self.sw[i] = self.beta * self.sw[i] + (1 - self.beta) * net._gw[i]**2
                self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * net._gb[i]**2
                net.weights[i] -= lr * net._gw[i] / (np.sqrt(self.sw[i]) + self.eps)
                net.biases[i] -= lr * net._gb[i] / (np.sqrt(self.sb[i]) + self.eps)

        elif self.name == 'adam':
            self.t += 1
            for i in range(len(net.weights)):
                self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * net._gw[i]
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * net._gb[i]
                self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * net._gw[i]**2
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * net._gb[i]**2
                mw_hat = self.mw[i] / (1 - self.beta1**self.t)
                mb_hat = self.mb[i] / (1 - self.beta1**self.t)
                vw_hat = self.vw[i] / (1 - self.beta2**self.t)
                vb_hat = self.vb[i] / (1 - self.beta2**self.t)
                net.weights[i] -= lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
                net.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


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


def nn_comparison(epochs=50, lr=0.5):
    """Train 4 identical networks with different optimizers."""
    X, labels = make_moons()
    base_net = MLP(seed=42)

    # Different LR for adaptive methods (they work best with smaller LR)
    lr_map = {'sgd': lr, 'momentum': lr, 'rmsprop': lr * 0.05, 'adam': lr * 0.05}

    trainers = {}
    for name in ['sgd', 'momentum', 'rmsprop', 'adam']:
        trainers[name] = MLPWithOptimizer(base_net.copy(), name)

    results = {name: [] for name in trainers}
    for epoch in range(epochs):
        for name, trainer in trainers.items():
            pred = trainer.net.forward(X)
            loss = trainer.net.loss(pred, labels)
            acc = trainer.net.accuracy(X, labels)
            trainer.net.backward(labels)
            trainer.update(lr_map[name])
            results[name].append({'epoch': epoch + 1, 'loss': loss, 'accuracy': acc})

    heatmaps = {name: trainer.net.predict_grid() for name, trainer in trainers.items()}
    pts = [{'x1': float(X[i, 0]), 'x2': float(X[i, 1]),
            'label': 'A' if labels[i, 0] > 0.5 else 'B'} for i in range(len(X))]

    return {'results': results, 'heatmaps': heatmaps, 'points': pts}


if __name__ == '__main__':
    print("=" * 55)
    print("  OPTIMIZERS COMPARISON")
    print("=" * 55)

    # 2D landscape
    r = run_optimizers('bowl', lr=0.01, steps=100)
    for name, path in r['paths'].items():
        print(f"  {name:10s}: final loss = {path[-1]['loss']:.6f}")

    # Neural network
    print("\n  Neural network (50 epochs):")
    r = nn_comparison(epochs=50)
    for name in ['sgd', 'momentum', 'rmsprop', 'adam']:
        last = r['results'][name][-1]
        print(f"  {name:10s}: loss={last['loss']:.4f}  acc={last['accuracy']:.0%}")
