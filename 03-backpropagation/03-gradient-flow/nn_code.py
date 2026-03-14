"""
=== MODULE 03.3: GRADIENT FLOW ===

In Module 03.2 you saw gradients get SMALLER as they flowed
backward through layers. This isn't a coincidence — it's the
central problem of deep learning.

THE VANISHING GRADIENT PROBLEM:

  Gradients are a PRODUCT of local derivatives along the chain:

    dL/dw1 = dL/dpred * dpred/dz_n * ... * dz_2/dh_1 * dh_1/dz_1 * dz_1/dw1

  For sigmoid, the maximum local gradient is 0.25 (at z=0).
  If you have 8 layers, the gradient passes through 8 sigmoids:

    0.25 * 0.25 * 0.25 * 0.25 * 0.25 * 0.25 * 0.25 * 0.25
    = 0.25^8
    = 0.0000153

  The gradient shrinks by 65,000x! Layer 1 barely learns at all.

THE EXPLODING GRADIENT PROBLEM:

  If local gradients are > 1 (from large weights), the product
  GROWS exponentially. Gradients become huge, weights blow up.

WHY RELU HELPS:

  ReLU's gradient is either 0 (dead) or 1 (pass-through).
  For active neurons, the gradient passes through UNCHANGED:

    1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 = 1

  No shrinkage! This is why ReLU replaced sigmoid in deep networks.
  (Dead neurons where gradient = 0 are a separate problem.)

THIS MODULE:
  Build a network from 2 to 10 layers deep.
  Watch gradient magnitude at each layer.
  Compare sigmoid vs ReLU — see why depth + sigmoid = disaster.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)


def simulate_gradient_flow(num_layers, activation='sigmoid', weight_scale=1.0, seed=42):
    """Simulate forward and backward pass through a deep narrow network.

    Architecture: 1 input → [hidden x num_layers] → 1 output
    Each hidden layer has 4 neurons.

    Returns gradient magnitude at each layer.
    """
    np.random.seed(seed)
    act_fn = sigmoid if activation == 'sigmoid' else relu
    act_grad_fn = sigmoid_grad if activation == 'sigmoid' else relu_grad

    hidden_size = 4
    x = np.array([1.0])
    y_true = 1.0

    # Build weights
    weights = []
    biases = []
    # Input → first hidden
    weights.append(np.random.randn(hidden_size, 1) * weight_scale)
    biases.append(np.zeros(hidden_size))
    # Hidden → hidden
    for _ in range(num_layers - 1):
        weights.append(np.random.randn(hidden_size, hidden_size) * weight_scale)
        biases.append(np.zeros(hidden_size))
    # Last hidden → output
    weights.append(np.random.randn(1, hidden_size) * weight_scale)
    biases.append(np.zeros(1))

    # === FORWARD PASS ===
    activations = [x]      # pre-activation z values not stored, we store post-activation
    pre_acts = []           # z values before activation
    h = x
    for i in range(len(weights)):
        z = weights[i] @ h + biases[i]
        pre_acts.append(z)
        h = act_fn(z)
        activations.append(h)

    pred = float(h[0])
    loss = (pred - y_true) ** 2

    # === BACKWARD PASS ===
    # Track gradient magnitude at each layer
    grad_magnitudes = []

    # Start: dL/dpred
    dL_dh = np.array([2 * (pred - y_true)])

    # Go backward through layers
    for i in range(len(weights) - 1, -1, -1):
        # Through activation
        local_grad = act_grad_fn(pre_acts[i])
        dL_dz = dL_dh * local_grad

        # Gradient magnitude at this layer
        grad_mag = float(np.mean(np.abs(dL_dz)))
        grad_magnitudes.append({
            'layer': i,
            'magnitude': grad_mag,
            'local_grad_mean': float(np.mean(np.abs(local_grad))),
        })

        # Gradient for weights at this layer
        # dL/dW = dL_dz @ h_prev^T
        # (not needed for visualization, just computing flow)

        # Pass gradient to previous layer
        dL_dh = weights[i].T @ dL_dz

    grad_magnitudes.reverse()  # layer 0 first

    # Compute the cumulative product of local gradients
    cumulative_product = 1.0
    products = []
    for g in grad_magnitudes:
        cumulative_product *= g['local_grad_mean']
        products.append(cumulative_product)

    return {
        'num_layers': num_layers,
        'activation': activation,
        'weight_scale': weight_scale,
        'pred': pred,
        'loss': loss,
        'layers': grad_magnitudes,
        'cumulative_products': products,
        'activations': [a.tolist() for a in activations],
    }


def compare_activations(num_layers, weight_scale=1.0):
    """Run both sigmoid and relu for comparison."""
    return {
        'sigmoid': simulate_gradient_flow(num_layers, 'sigmoid', weight_scale),
        'relu': simulate_gradient_flow(num_layers, 'relu', weight_scale),
    }


def depth_sweep(activation='sigmoid', weight_scale=1.0, max_depth=10):
    """How does gradient at layer 0 change as we add more layers?"""
    results = []
    for d in range(2, max_depth + 1):
        r = simulate_gradient_flow(d, activation, weight_scale)
        first_layer_grad = r['layers'][0]['magnitude']
        results.append({
            'depth': d,
            'first_layer_grad': first_layer_grad,
        })
    return results


if __name__ == '__main__':
    print("=" * 55)
    print("  GRADIENT FLOW — Vanishing & Exploding Gradients")
    print("=" * 55)

    for act in ['sigmoid', 'relu']:
        print(f"\n  {act.upper()} — 8 layers:")
        r = simulate_gradient_flow(8, act)
        for g in r['layers']:
            bar = '#' * max(1, int(g['magnitude'] * 200))
            print(f"    Layer {g['layer']:2d}: |grad|={g['magnitude']:.6f}  {bar}")

    print("\n  Sigmoid gradient at layer 0 vs depth:")
    for item in depth_sweep('sigmoid'):
        print(f"    depth={item['depth']:2d}  grad={item['first_layer_grad']:.8f}")
