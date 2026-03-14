"""
=== MODULE 02.4: A NEURON LEARNS ===

This is the payoff. Everything we've built comes together:

  Module 01.1: A neuron computes  weighted_sum = w1*x1 + w2*x2 + bias
  Module 01.2: Then applies       output = sigmoid(weighted_sum)
  Module 01.3: The output decides  Class A (>= 0.5) or Class B (< 0.5)
  Module 02.1: A loss function     measures how wrong the prediction is
  Module 02.2: Gradient descent    adjusts the weights to reduce the loss
  Module 02.3: Learning rate       controls how big each adjustment is

THE TRAINING LOOP (the heart of all machine learning):
  1. FORWARD PASS — feed data through the neuron, get predictions
  2. COMPUTE LOSS — how wrong were the predictions?
  3. COMPUTE GRADIENTS — which direction should each weight move?
  4. UPDATE WEIGHTS — nudge weights in the right direction
  5. REPEAT until the loss is small

THE GRADIENT (for a single neuron with sigmoid + cross-entropy):

  The math works out beautifully:
    error = prediction - true_label     (that's it!)
    dL/dw1 = error * x1                 (how much w1 should change)
    dL/dw2 = error * x2                 (how much w2 should change)
    dL/db  = error                       (how much bias should change)

  This is the simplest possible case. In Module 03 we'll see how
  this generalizes to multi-layer networks (backpropagation).
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class TrainableNeuron:
    """A neuron that can learn from labeled data.

    Combines everything: forward pass, loss, gradients, weight updates.
    """

    def __init__(self):
        self.w1 = 0.0
        self.w2 = 0.0
        self.bias = 0.0
        self.history = []  # training history for plotting

    def reset(self, w1=0.0, w2=0.0, bias=0.0):
        """Reset to initial state."""
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.history = []

    def get_state(self):
        return {
            'w1': self.w1,
            'w2': self.w2,
            'bias': self.bias,
            'step': len(self.history),
        }

    def predict(self, X):
        """Forward pass: compute predictions for all data points.

        X: numpy array of shape (N, 2) — the input data
        Returns: numpy array of shape (N,) — probabilities
        """
        z = X[:, 0] * self.w1 + X[:, 1] * self.w2 + self.bias
        return sigmoid(z)

    def compute_loss(self, predictions, labels):
        """Binary cross-entropy loss averaged over all samples."""
        eps = 1e-7
        p = np.clip(predictions, eps, 1 - eps)
        loss = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        return float(loss)

    def train_step(self, X, labels, lr):
        """One complete training step: forward, loss, gradients, update.

        X: shape (N, 2)
        labels: shape (N,) — 1.0 for class A, 0.0 for class B
        lr: learning rate

        Returns dict with all details for visualization.
        """
        N = len(labels)

        # === STEP 1: FORWARD PASS ===
        predictions = self.predict(X)

        # === STEP 2: COMPUTE LOSS ===
        loss = self.compute_loss(predictions, labels)

        # === STEP 3: COMPUTE GRADIENTS ===
        # The beautiful simplicity of sigmoid + BCE:
        errors = predictions - labels  # shape (N,)

        # Average gradient over all data points
        grad_w1 = float(np.mean(errors * X[:, 0]))
        grad_w2 = float(np.mean(errors * X[:, 1]))
        grad_b = float(np.mean(errors))

        # === STEP 4: UPDATE WEIGHTS ===
        self.w1 -= lr * grad_w1
        self.w2 -= lr * grad_w2
        self.bias -= lr * grad_b

        # Compute new loss after update
        new_predictions = self.predict(X)
        new_loss = self.compute_loss(new_predictions, labels)

        # Accuracy
        correct = np.sum((new_predictions >= 0.5) == (labels >= 0.5))
        accuracy = float(correct / N)

        # Record history
        step_data = {
            'step': len(self.history),
            'loss_before': loss,
            'loss_after': new_loss,
            'accuracy': accuracy,
            'w1': self.w1,
            'w2': self.w2,
            'bias': self.bias,
            'grad_w1': grad_w1,
            'grad_w2': grad_w2,
            'grad_b': grad_b,
        }
        self.history.append(step_data)

        return step_data

    def train_many(self, X, labels, lr, num_steps):
        """Run multiple training steps."""
        results = []
        for _ in range(num_steps):
            r = self.train_step(X, labels, lr)
            results.append(r)
        return results

    def get_decision_boundary(self, x_min=-4, x_max=4):
        """Compute decision boundary line for visualization."""
        if abs(self.w2) > 1e-8:
            slope = -self.w1 / self.w2
            intercept = -self.bias / self.w2
            return {
                'type': 'line',
                'x1': [x_min, x_max],
                'x2': [slope * x_min + intercept, slope * x_max + intercept],
            }
        elif abs(self.w1) > 1e-8:
            return {'type': 'vertical', 'x1_val': -self.bias / self.w1}
        return {'type': 'none'}

    def get_heatmap(self, x_min=-4, x_max=4, resolution=50):
        """Prediction heatmap for the full space."""
        x1v = np.linspace(x_min, x_max, resolution)
        x2v = np.linspace(x_min, x_max, resolution)
        X1, X2 = np.meshgrid(x1v, x2v)
        Z = X1 * self.w1 + X2 * self.w2 + self.bias
        probs = sigmoid(Z)
        return {'probs': probs.tolist(), 'x1v': x1v.tolist(), 'x2v': x2v.tolist()}


# ============================================================
# DATASETS
# ============================================================
def make_dataset(name):
    np.random.seed(42)
    if name == 'simple':
        a = np.random.randn(20, 2) * 0.7 + [1.5, 1.5]
        b = np.random.randn(20, 2) * 0.7 + [-1.5, -1.5]
    elif name == 'horizontal':
        a = np.random.randn(20, 2) * 0.6 + [0, 1.5]
        b = np.random.randn(20, 2) * 0.6 + [0, -1.5]
    elif name == 'diagonal':
        a = np.random.randn(20, 2) * 0.5 + [2, -1]
        b = np.random.randn(20, 2) * 0.5 + [-1, 2]
    elif name == 'messy':
        a = np.random.randn(25, 2) * 1.2 + [0.8, 0.8]
        b = np.random.randn(25, 2) * 1.2 + [-0.8, -0.8]
    else:
        a = np.random.randn(15, 2)
        b = np.random.randn(15, 2) + 2

    X = np.vstack([a, b])
    labels = np.array([1.0] * len(a) + [0.0] * len(b))
    points = []
    for i in range(len(X)):
        points.append({'x1': round(float(X[i, 0]), 3), 'x2': round(float(X[i, 1]), 3),
                        'label': 'A' if labels[i] > 0.5 else 'B'})
    return {'points': points, 'X': X.tolist(), 'labels': labels.tolist()}


DATASETS = {
    'simple': 'Two easy clusters — a warm-up',
    'horizontal': 'Split by a horizontal line',
    'diagonal': 'Split along the diagonal',
    'messy': 'Overlapping clusters — imperfect separation',
}


if __name__ == '__main__':
    print("=" * 55)
    print("  A NEURON LEARNS")
    print("=" * 55)

    ds = make_dataset('simple')
    X = np.array(ds['X'])
    labels = np.array(ds['labels'])

    neuron = TrainableNeuron()
    print(f"\n  Before training: w1={neuron.w1:.2f} w2={neuron.w2:.2f} b={neuron.bias:.2f}")

    for i in range(100):
        r = neuron.train_step(X, labels, lr=1.0)
        if i < 5 or i % 20 == 19:
            print(f"  Step {r['step']:3d}: loss={r['loss_after']:.4f}  acc={r['accuracy']:.0%}  "
                  f"w1={r['w1']:.3f} w2={r['w2']:.3f} b={r['bias']:.3f}")

    print(f"\n  After training: loss={r['loss_after']:.4f}  accuracy={r['accuracy']:.0%}")
