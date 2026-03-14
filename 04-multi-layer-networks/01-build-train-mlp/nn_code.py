"""
=== MODULE 04: BUILD & TRAIN AN MLP ===

Everything from Phases 1 comes together in one interactive playground.

You now have all the building blocks:
  - Neurons compute weighted sums + activation (Module 01)
  - Loss functions measure error (Module 02.1)
  - Gradient descent minimizes loss (Module 02.2-2.3)
  - Backpropagation computes gradients through layers (Module 03)

A MULTI-LAYER PERCEPTRON (MLP) stacks these into a powerful network:

    Input → [Hidden Layer 1] → [Hidden Layer 2] → ... → Output

KEY IDEAS THIS MODULE DEMONSTRATES:

  1. LAYERS: Adding hidden layers lets the network learn nonlinear patterns.
     A single neuron can only draw a straight line (Module 01.3).
     One hidden layer can draw curves. Multiple layers can draw anything.

  2. NEURONS PER LAYER: More neurons = more "kinks" in the decision boundary.
     This is the universal approximation from our FEM/basis-function discussion.

  3. ACTIVATION FUNCTIONS: ReLU creates piecewise-linear boundaries (sharp angles).
     Sigmoid/Tanh create smooth curves. ReLU trains faster (Module 03.3).

  4. UNIVERSAL APPROXIMATION: With enough neurons in even one hidden layer,
     an MLP can approximate ANY decision boundary to any accuracy.
     Add more neurons and watch the boundary get finer — just like refining a mesh.
"""

import numpy as np


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(a):
    """Gradient given the activation value (not z)."""
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_grad_from_z(z):
    return (z > 0).astype(float)

def tanh_fn(z):
    return np.tanh(z)

def tanh_grad(a):
    return 1 - a**2

ACTIVATIONS = {
    'relu':    (relu, relu_grad_from_z, 'from_z'),
    'sigmoid': (sigmoid, sigmoid_grad, 'from_a'),
    'tanh':    (tanh_fn, tanh_grad, 'from_a'),
}


# ============================================================
# THE MLP CLASS
# ============================================================
class MLP:
    """A fully-connected multi-layer perceptron, built from scratch.

    Architecture is defined by layer_sizes, e.g.:
      [2, 8, 4, 1] = 2 inputs, 8 hidden, 4 hidden, 1 output
    """

    def __init__(self, layer_sizes, activation='relu', seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.act_fn, self.act_grad, self.grad_mode = ACTIVATIONS[activation]

        # Initialize weights with He initialization (good for ReLU)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(np.random.randn(layer_sizes[i+1], fan_in) * scale)
            self.biases.append(np.zeros(layer_sizes[i+1]))

        self.train_history = []

    def forward(self, X):
        """Forward pass. X shape: (N, input_dim).
        Returns prediction and cached values for backprop."""
        self.cache = {'a': [X]}  # activations at each layer (a[0] = input)
        self.cache['z'] = []     # pre-activations

        h = X
        for i in range(len(self.weights)):
            z = h @ self.weights[i].T + self.biases[i]  # (N, layer_size)
            self.cache['z'].append(z)

            if i == len(self.weights) - 1:
                # Output layer: always sigmoid for binary classification
                h = sigmoid(z)
            else:
                # Hidden layers: use chosen activation
                h = self.act_fn(z)
            self.cache['a'].append(h)

        return h  # shape (N, 1) or (N, output_dim)

    def compute_loss(self, predictions, labels):
        """Binary cross-entropy loss."""
        eps = 1e-7
        p = np.clip(predictions, eps, 1 - eps)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def backward(self, labels):
        """Backward pass. Computes gradients for all weights and biases."""
        N = labels.shape[0]
        predictions = self.cache['a'][-1]

        # Gradient at output (sigmoid + BCE simplifies to pred - label)
        delta = (predictions - labels) / N  # (N, 1)

        self.grad_w = []
        self.grad_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients for this layer's weights and biases
            a_prev = self.cache['a'][i]  # (N, prev_size)
            dW = delta.T @ a_prev       # (layer_size, prev_size)
            db = np.sum(delta, axis=0)   # (layer_size,)
            self.grad_w.insert(0, dW)
            self.grad_b.insert(0, db)

            if i > 0:
                # Propagate gradient to previous layer
                delta = delta @ self.weights[i]  # (N, prev_size)
                # Through activation
                if self.grad_mode == 'from_z':
                    delta = delta * self.act_grad(self.cache['z'][i-1])
                else:
                    delta = delta * self.act_grad(self.cache['a'][i])

    def train_step(self, X, labels, lr):
        """One complete training step."""
        predictions = self.forward(X)
        loss = self.compute_loss(predictions, labels)

        correct = np.sum((predictions >= 0.5).flatten() == (labels >= 0.5).flatten())
        accuracy = float(correct / len(labels))

        self.backward(labels)

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.grad_w[i]
            self.biases[i] -= lr * self.grad_b[i]

        self.train_history.append({'loss': loss, 'accuracy': accuracy})
        return {'loss': loss, 'accuracy': accuracy, 'step': len(self.train_history)}

    def train_many(self, X, labels, lr, steps):
        """Multiple training steps."""
        results = []
        for _ in range(steps):
            r = self.train_step(X, labels, lr)
            results.append(r)
        return results

    def predict_grid(self, x_min=-3, x_max=3, resolution=60):
        """Prediction heatmap for visualization."""
        x1 = np.linspace(x_min, x_max, resolution)
        x2 = np.linspace(x_min, x_max, resolution)
        X1, X2 = np.meshgrid(x1, x2)
        grid_X = np.column_stack([X1.ravel(), X2.ravel()])
        preds = self.forward(grid_X)
        return preds.reshape(resolution, resolution).tolist()

    def get_info(self):
        """Summary of the network."""
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        return {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'total_params': total_params,
            'num_layers': len(self.weights),
        }


# ============================================================
# DATASETS
# ============================================================
def make_dataset(name, n=200, noise=0.1, seed=42):
    """Generate 2D classification datasets."""
    np.random.seed(seed)

    if name == 'linear':
        X = np.random.randn(n, 2)
        labels = (X[:, 0] + X[:, 1] > 0).astype(float)

    elif name == 'xor':
        X = np.random.randn(n, 2) * 0.8
        labels = ((X[:, 0] * X[:, 1]) > 0).astype(float)

    elif name == 'circle':
        r = np.random.uniform(0, 3, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        labels = (r > 1.5).astype(float)
        X += np.random.randn(n, 2) * noise

    elif name == 'moons':
        # Two half-moons
        n_half = n // 2
        theta1 = np.linspace(0, np.pi, n_half)
        theta2 = np.linspace(0, np.pi, n - n_half)
        x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
        x2 = np.column_stack([np.cos(theta2) + 0.5, -np.sin(theta2) + 0.5])
        X = np.vstack([x1, x2]) + np.random.randn(n, 2) * noise
        labels = np.array([1.0] * n_half + [0.0] * (n - n_half))

    elif name == 'spiral':
        n_half = n // 2
        theta1 = np.linspace(0, 3 * np.pi, n_half)
        r1 = np.linspace(0.3, 2.5, n_half)
        theta2 = np.linspace(0, 3 * np.pi, n - n_half) + np.pi
        r2 = np.linspace(0.3, 2.5, n - n_half)
        x1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
        x2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        X = np.vstack([x1, x2]) + np.random.randn(n, 2) * noise * 2
        labels = np.array([1.0] * n_half + [0.0] * (n - n_half))

    else:
        X = np.random.randn(n, 2)
        labels = (X[:, 0] > 0).astype(float)

    # Shuffle
    idx = np.random.permutation(n)
    X = X[idx]
    labels = labels[idx]

    return {
        'X': X.tolist(),
        'labels': labels.tolist(),
        'points': [{'x1': float(X[i, 0]), 'x2': float(X[i, 1]),
                     'label': 'A' if labels[i] > 0.5 else 'B'} for i in range(n)],
    }


DATASETS = {
    'linear':  'Linearly separable — one neuron can do this',
    'xor':     'XOR pattern — needs at least one hidden layer',
    'circle':  'Ring around center — needs nonlinear boundary',
    'moons':   'Two crescent moons — moderate difficulty',
    'spiral':  'Two interleaved spirals — needs depth and width!',
}


if __name__ == '__main__':
    print("=" * 55)
    print("  MLP PLAYGROUND")
    print("=" * 55)

    ds = make_dataset('spiral')
    X = np.array(ds['X'])
    labels = np.array(ds['labels']).reshape(-1, 1)

    mlp = MLP([2, 16, 8, 1], activation='relu', seed=42)
    print(f"  Architecture: {mlp.layer_sizes}")
    print(f"  Parameters: {mlp.get_info()['total_params']}")

    for i in range(200):
        r = mlp.train_step(X, labels, lr=0.5)
        if i < 5 or i % 50 == 49:
            print(f"  Step {r['step']:3d}: loss={r['loss']:.4f}  acc={r['accuracy']:.0%}")
