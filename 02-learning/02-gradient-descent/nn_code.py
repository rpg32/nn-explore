"""
=== MODULE 02.2: GRADIENT DESCENT ===

Now we know:
  - A neuron computes outputs (Module 01)
  - A loss function measures how wrong those outputs are (Module 02.1)

The missing piece: HOW DOES THE NEURON IMPROVE?

Answer: Gradient Descent. The most important algorithm in all of
machine learning.

THE INTUITION — Rolling downhill:
  Imagine the loss as a hilly landscape. The height at any point
  is the loss value for those particular weight settings. We want
  to find the LOWEST point (minimum loss = best weights).

  Gradient descent is blindfolded hill-walking:
    1. Feel which direction is steepest downhill (the gradient)
    2. Take a step in that direction
    3. Repeat until you reach the bottom

THE MATH:
  The GRADIENT is the vector of partial derivatives — it tells you
  the slope of the loss in each weight direction.

  For a loss L(w1, w2):
    gradient = [ dL/dw1, dL/dw2 ]

  The gradient points UPHILL (direction of steepest increase).
  So we go the OPPOSITE way:

    w1_new = w1 - learning_rate * dL/dw1
    w2_new = w2 - learning_rate * dL/dw2

  That's it. That's the entire algorithm.

LEARNING RATE:
  The learning rate controls step size:
    - Too small → crawls painfully slowly
    - Too large → overshoots, bounces around, may never converge
    - Just right → smooth descent to the minimum

LANDSCAPES:
  Different weight configurations create different loss landscapes:
    - "Bowl" — simple, one clear minimum (easy)
    - "Elongated" — stretched in one direction (causes zigzagging)
    - "Saddle" — has flat regions that slow down descent
"""

import numpy as np


# ============================================================
# LOSS LANDSCAPES — different shapes to explore
# ============================================================

LANDSCAPES = {
    'bowl': {
        'label': 'Simple Bowl',
        'desc': 'A symmetric bowl — the easiest landscape. One clear minimum.',
        'fn': lambda w1, w2: w1**2 + w2**2,
        'grad': lambda w1, w2: (2*w1, 2*w2),
        'minimum': (0.0, 0.0),
    },
    'elongated': {
        'label': 'Elongated Valley',
        'desc': 'Stretched 10x in one direction — watch the zigzagging!',
        'fn': lambda w1, w2: w1**2 + 10*w2**2,
        'grad': lambda w1, w2: (2*w1, 20*w2),
        'minimum': (0.0, 0.0),
    },
    'offset': {
        'label': 'Off-Center Bowl',
        'desc': 'Minimum is not at the origin — the bias must shift.',
        'fn': lambda w1, w2: (w1 - 2)**2 + (w2 + 1)**2,
        'grad': lambda w1, w2: (2*(w1 - 2), 2*(w2 + 1)),
        'minimum': (2.0, -1.0),
    },
    'saddle': {
        'label': 'Saddle + Bowl',
        'desc': 'A saddle region that can trap naive descent.',
        'fn': lambda w1, w2: w1**2 - 0.5*w1*w2 + w2**2 + 0.1*(w1 + w2),
        'grad': lambda w1, w2: (2*w1 - 0.5*w2 + 0.1, 2*w2 - 0.5*w1 + 0.1),
        'minimum': (-0.1, -0.1),  # approximate
    },
}


def compute_landscape(name, w_min=-4, w_max=4, resolution=80):
    """Compute the loss surface on a grid for visualization."""
    info = LANDSCAPES[name]
    w1_vals = np.linspace(w_min, w_max, resolution)
    w2_vals = np.linspace(w_min, w_max, resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    L = info['fn'](W1, W2)

    return {
        'w1_vals': w1_vals.tolist(),
        'w2_vals': w2_vals.tolist(),
        'loss': L.tolist(),
        'loss_min': float(L.min()),
        'loss_max': float(L.max()),
        'label': info['label'],
        'desc': info['desc'],
        'minimum': list(info['minimum']),
    }


def gradient_descent_step(name, w1, w2, lr):
    """Perform ONE step of gradient descent.

    Returns the new position, the gradient, and the loss before/after.
    """
    info = LANDSCAPES[name]

    # Current loss
    loss_before = float(info['fn'](w1, w2))

    # Compute gradient (slope in each direction)
    dw1, dw2 = info['grad'](w1, w2)

    # Clip gradient magnitude to prevent catastrophic steps
    grad_mag = np.sqrt(dw1**2 + dw2**2)
    max_grad = 50.0
    if grad_mag > max_grad:
        scale = max_grad / grad_mag
        dw1 *= scale
        dw2 *= scale

    # Update: step in the OPPOSITE direction of the gradient
    w1_new = w1 - lr * dw1
    w2_new = w2 - lr * dw2

    # New loss
    loss_after = float(info['fn'](w1_new, w2_new))

    return {
        'w1': float(w1_new),
        'w2': float(w2_new),
        'w1_old': float(w1),
        'w2_old': float(w2),
        'grad_w1': float(dw1),
        'grad_w2': float(dw2),
        'grad_magnitude': float(np.sqrt(dw1**2 + dw2**2)),
        'lr': float(lr),
        'loss_before': loss_before,
        'loss_after': loss_after,
        'improved': loss_after < loss_before,
    }


def run_descent(name, w1_start, w2_start, lr, num_steps):
    """Run multiple steps of gradient descent, recording the path."""
    info = LANDSCAPES[name]
    w1, w2 = w1_start, w2_start

    path = [{
        'step': 0,
        'w1': float(w1),
        'w2': float(w2),
        'loss': float(info['fn'](w1, w2)),
    }]

    for i in range(num_steps):
        result = gradient_descent_step(name, w1, w2, lr)
        w1, w2 = result['w1'], result['w2']

        # Clamp to prevent explosion
        w1 = max(-10, min(10, w1))
        w2 = max(-10, min(10, w2))

        loss = float(info['fn'](w1, w2))
        path.append({
            'step': i + 1,
            'w1': float(w1),
            'w2': float(w2),
            'loss': loss,
            'grad_magnitude': result['grad_magnitude'],
        })

        # Early stop: if loss is exploding, mark it and bail out
        if loss > 1e6 or np.isnan(loss) or np.isinf(loss):
            path[-1]['diverged'] = True
            break

    return {
        'path': path,
        'final_loss': path[-1]['loss'],
        'final_w1': path[-1]['w1'],
        'final_w2': path[-1]['w2'],
        'minimum': list(info['minimum']),
    }


# ============================================================
# TRY IT YOURSELF
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  GRADIENT DESCENT — Rolling Downhill")
    print("=" * 55)
    print()

    # Simple bowl landscape
    name = 'bowl'
    result = run_descent(name, w1_start=3.5, w2_start=-3.0, lr=0.1, num_steps=20)

    print(f"  Landscape: {LANDSCAPES[name]['label']}")
    print(f"  Start: w1=3.50, w2=-3.00, loss={result['path'][0]['loss']:.4f}")
    print(f"  Learning rate: 0.1")
    print()

    for p in result['path'][:8]:
        print(f"  Step {p['step']:2d}: w1={p['w1']:+.4f}  w2={p['w2']:+.4f}  loss={p['loss']:.6f}")

    print(f"  ...")
    p = result['path'][-1]
    print(f"  Step {p['step']:2d}: w1={p['w1']:+.4f}  w2={p['w2']:+.4f}  loss={p['loss']:.6f}")
    print(f"\n  Minimum at: {result['minimum']}")
    print(f"  Final loss: {result['final_loss']:.8f}")
