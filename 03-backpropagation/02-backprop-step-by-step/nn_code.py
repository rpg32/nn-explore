"""
=== MODULE 03.2: BACKPROPAGATION STEP-BY-STEP ===

In Module 03.1 we saw the chain rule through a single neuron.
Now we apply it to a REAL multi-layer network:

    Input (2) → Hidden layer (2 neurons) → Output (1 neuron) → Loss

This tiny network has:
  - 4 weights in the hidden layer (2 inputs × 2 neurons)
  - 2 biases in the hidden layer
  - 2 weights in the output layer (2 hidden → 1 output)
  - 1 bias in the output layer
  = 9 learnable parameters total

BACKPROPAGATION walks through two phases:

  FORWARD PASS (left to right):
    1. Compute hidden layer activations: h = sigmoid(W1 @ x + b1)
    2. Compute output: pred = sigmoid(W2 @ h + b2)
    3. Compute loss: L = (pred - y_true)^2

  BACKWARD PASS (right to left):
    4. Gradient at output: dL/dpred
    5. Gradient through output sigmoid: dL/dz2
    6. Gradients for output weights: dL/dW2, dL/db2
    7. Gradient passed to hidden layer: dL/dh
    8. Gradient through hidden sigmoids: dL/dz1
    9. Gradients for hidden weights: dL/dW1, dL/db1

  UPDATE (all at once):
    10. Subtract lr * gradient from every weight and bias

The key insight: we compute gradients for ALL layers in one
backward sweep. Each layer only needs the gradient from the
layer above it (passed down via the chain rule).
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class TinyNetwork:
    """A minimal 2-layer network for demonstrating backprop.

    Architecture: 2 inputs → 2 hidden (sigmoid) → 1 output (sigmoid)
    """

    def __init__(self):
        np.random.seed(42)
        # Hidden layer: 2 inputs → 2 hidden neurons
        self.W1 = np.array([[0.4, -0.5],
                            [0.3,  0.6]])   # shape (2, 2)
        self.b1 = np.array([0.1, -0.1])     # shape (2,)

        # Output layer: 2 hidden → 1 output
        self.W2 = np.array([[0.7, -0.4]])    # shape (1, 2)
        self.b2 = np.array([0.05])           # shape (1,)

    def reset(self):
        self.__init__()

    def get_params(self):
        return {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
        }

    def forward_detailed(self, x, y_true):
        """Full forward pass with every intermediate value recorded."""
        x = np.array(x)

        # Hidden layer
        z1 = self.W1 @ x + self.b1           # (2,)
        h = sigmoid(z1)                        # (2,)

        # Output layer
        z2 = self.W2 @ h + self.b2            # (1,)
        pred = float(sigmoid(z2[0]))

        # Loss
        loss = (pred - y_true) ** 2

        return {
            'x': x.tolist(),
            'y_true': y_true,
            'z1': z1.tolist(),          # pre-activation hidden
            'h': h.tolist(),            # hidden activations
            'z2': float(z2[0]),         # pre-activation output
            'pred': pred,               # output
            'loss': float(loss),
        }

    def backward_detailed(self, x, y_true):
        """Full forward + backward pass with all gradients."""
        x = np.array(x)

        # === FORWARD ===
        z1 = self.W1 @ x + self.b1
        h = sigmoid(z1)
        z2_val = float((self.W2 @ h + self.b2)[0])
        pred = float(sigmoid(z2_val))
        loss = (pred - y_true) ** 2

        # === BACKWARD ===
        # Step 1: dL/dpred = 2 * (pred - y_true)
        dL_dpred = 2 * (pred - y_true)

        # Step 2: dL/dz2 = dL/dpred * sigmoid'(z2)
        sig_prime_z2 = pred * (1 - pred)
        dL_dz2 = dL_dpred * sig_prime_z2

        # Step 3: dL/dW2 and dL/db2
        dL_dW2 = [dL_dz2 * h[0], dL_dz2 * h[1]]
        dL_db2 = dL_dz2

        # Step 4: dL/dh (gradient passed DOWN to hidden layer)
        dL_dh = [dL_dz2 * self.W2[0, 0], dL_dz2 * self.W2[0, 1]]

        # Step 5: dL/dz1 = dL/dh * sigmoid'(z1)
        sig_prime_z1 = [h[0] * (1 - h[0]), h[1] * (1 - h[1])]
        dL_dz1 = [dL_dh[0] * sig_prime_z1[0], dL_dh[1] * sig_prime_z1[1]]

        # Step 6: dL/dW1 and dL/db1
        dL_dW1 = [
            [dL_dz1[0] * x[0], dL_dz1[0] * x[1]],
            [dL_dz1[1] * x[0], dL_dz1[1] * x[1]],
        ]
        dL_db1 = dL_dz1

        return {
            'forward': {
                'x': x.tolist(), 'y_true': y_true,
                'z1': z1.tolist(), 'h': h.tolist(),
                'z2': z2_val, 'pred': pred, 'loss': loss,
            },
            'backward': {
                'dL_dpred': dL_dpred,
                'sig_prime_z2': sig_prime_z2,
                'dL_dz2': dL_dz2,
                'dL_dW2': dL_dW2,
                'dL_db2': dL_db2,
                'dL_dh': dL_dh,
                'sig_prime_z1': sig_prime_z1,
                'dL_dz1': dL_dz1,
                'dL_dW1': dL_dW1,
                'dL_db1': dL_db1,
            },
            'params': self.get_params(),
        }

    def train_step(self, x, y_true, lr):
        """One complete training step with full detail."""
        result = self.backward_detailed(x, y_true)
        b = result['backward']

        # Update weights
        self.W2[0, 0] -= lr * b['dL_dW2'][0]
        self.W2[0, 1] -= lr * b['dL_dW2'][1]
        self.b2[0] -= lr * b['dL_db2']

        self.W1[0, 0] -= lr * b['dL_dW1'][0][0]
        self.W1[0, 1] -= lr * b['dL_dW1'][0][1]
        self.W1[1, 0] -= lr * b['dL_dW1'][1][0]
        self.W1[1, 1] -= lr * b['dL_dW1'][1][1]
        self.b1[0] -= lr * b['dL_db1'][0]
        self.b1[1] -= lr * b['dL_db1'][1]

        result['new_params'] = self.get_params()
        # Compute new loss
        fwd_new = self.forward_detailed(x, y_true)
        result['new_loss'] = fwd_new['loss']
        result['new_pred'] = fwd_new['pred']
        return result

    def train_many(self, x, y_true, lr, steps):
        """Multiple training steps, return loss history."""
        history = []
        for i in range(steps):
            r = self.train_step(x, y_true, lr)
            history.append({
                'step': i,
                'loss': r['forward']['loss'],
                'pred': r['forward']['pred'],
                'new_loss': r['new_loss'],
            })
        return history


if __name__ == '__main__':
    net = TinyNetwork()
    x = [1.0, 0.5]
    y = 1.0

    print("=" * 55)
    print("  BACKPROP STEP-BY-STEP")
    print("=" * 55)

    r = net.backward_detailed(x, y)
    f = r['forward']
    b = r['backward']

    print(f"\n  FORWARD:")
    print(f"    x = {f['x']}")
    print(f"    z1 = {[round(v,4) for v in f['z1']]}")
    print(f"    h  = {[round(v,4) for v in f['h']]}")
    print(f"    z2 = {f['z2']:.4f}")
    print(f"    pred = {f['pred']:.4f}")
    print(f"    loss = {f['loss']:.4f}")

    print(f"\n  BACKWARD:")
    print(f"    dL/dpred = {b['dL_dpred']:.4f}")
    print(f"    dL/dz2   = {b['dL_dz2']:.4f}")
    print(f"    dL/dh    = {[round(v,4) for v in b['dL_dh']]}")
    print(f"    dL/dz1   = {[round(v,4) for v in b['dL_dz1']]}")
    print(f"    dL/dW2   = {[round(v,4) for v in b['dL_dW2']]}")
    print(f"    dL/dW1   = {[[round(v,4) for v in row] for row in b['dL_dW1']]}")
