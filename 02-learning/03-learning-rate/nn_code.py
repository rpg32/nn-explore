"""
=== MODULE 02.3: LEARNING RATE ===

You already saw in Module 02.2 that the learning rate matters.
Now let's nail down exactly WHY and HOW.

The learning rate (lr) is the SINGLE MOST IMPORTANT hyperparameter
in training a neural network.

    w_new = w_old - lr * gradient

It controls step size:

  TOO SMALL (e.g., 0.001):
    - Each step barely moves
    - Training takes forever
    - May get stuck in shallow local minima
    - But at least it's stable!

  TOO LARGE (e.g., 1.0):
    - Steps are huge
    - Overshoots the minimum, bounces around
    - Loss goes UP instead of down
    - May diverge to infinity (as you saw!)

  JUST RIGHT (the "Goldilocks zone"):
    - Steady descent toward the minimum
    - Converges in a reasonable number of steps
    - The "right" value depends on the landscape shape

THE KEY INSIGHT:
  There is no universal "best" learning rate. It depends on:
  - The loss landscape shape (steep vs flat)
  - The current position (near minimum vs far away)
  - The batch size (bigger batches = can use bigger lr)

  This is why modern optimizers (Adam, etc.) ADAPT the learning
  rate automatically. More on that in Module 05.

THIS MODULE:
  Run the same descent with 3 different learning rates
  simultaneously. See exactly how they compare.
"""

import numpy as np


def loss_fn(w1, w2):
    """A moderately elongated bowl — interesting but not extreme."""
    return (w1 - 1)**2 + 3 * (w2 + 0.5)**2


def grad_fn(w1, w2):
    """Gradient of the loss function."""
    return (2 * (w1 - 1), 6 * (w2 + 0.5))


MINIMUM = (1.0, -0.5)


def run_three(w1_start, w2_start, lr_slow, lr_good, lr_fast, num_steps=60):
    """Run gradient descent with three different learning rates from the same start.

    Returns three paths for comparison.
    """
    results = {}
    for label, lr in [('slow', lr_slow), ('good', lr_good), ('fast', lr_fast)]:
        w1, w2 = w1_start, w2_start
        path = [{'step': 0, 'w1': w1, 'w2': w2, 'loss': float(loss_fn(w1, w2))}]

        for i in range(num_steps):
            dw1, dw2 = grad_fn(w1, w2)

            # Gradient clipping
            mag = np.sqrt(dw1**2 + dw2**2)
            if mag > 50:
                dw1 *= 50 / mag
                dw2 *= 50 / mag

            w1 = w1 - lr * dw1
            w2 = w2 - lr * dw2

            # Clamp
            w1 = max(-8, min(8, w1))
            w2 = max(-8, min(8, w2))

            loss = float(loss_fn(w1, w2))
            path.append({'step': i + 1, 'w1': w1, 'w2': w2, 'loss': loss})

            if loss > 1e5 or np.isnan(loss):
                path[-1]['diverged'] = True
                break

        results[label] = {
            'lr': lr,
            'path': path,
            'final_loss': path[-1]['loss'],
            'steps': len(path) - 1,
        }

    return results


def compute_landscape(w_min=-4, w_max=6, resolution=80):
    """Compute the loss surface for visualization."""
    w1v = np.linspace(w_min, w_max, resolution)
    w2v = np.linspace(w_min, w_max, resolution)
    W1, W2 = np.meshgrid(w1v, w2v)
    L = loss_fn(W1, W2)
    return {
        'w1_vals': w1v.tolist(),
        'w2_vals': w2v.tolist(),
        'loss': L.tolist(),
        'loss_min': float(L.min()),
        'loss_max': float(L.max()),
        'minimum': list(MINIMUM),
        'w_min': w_min,
        'w_max': w_max,
    }


def lr_sweep(w1_start, w2_start, num_steps=40, num_lrs=50):
    """Try many learning rates, report the final loss for each.

    This shows the "sweet spot" — a curve of final loss vs learning rate.
    """
    lrs = np.logspace(-3, 0.3, num_lrs)  # 0.001 to ~2.0
    results = []
    for lr in lrs:
        w1, w2 = w1_start, w2_start
        for _ in range(num_steps):
            dw1, dw2 = grad_fn(w1, w2)
            mag = np.sqrt(dw1**2 + dw2**2)
            if mag > 50:
                dw1 *= 50 / mag
                dw2 *= 50 / mag
            w1 = max(-8, min(8, w1 - lr * dw1))
            w2 = max(-8, min(8, w2 - lr * dw2))
        loss = float(loss_fn(w1, w2))
        results.append({'lr': float(lr), 'loss': min(loss, 200)})
    return results


if __name__ == '__main__':
    print("=" * 55)
    print("  LEARNING RATE COMPARISON")
    print("=" * 55)
    print()

    res = run_three(3.5, 3.0, lr_slow=0.005, lr_good=0.1, lr_fast=0.8, num_steps=30)
    for label in ['slow', 'good', 'fast']:
        r = res[label]
        print(f"  lr={r['lr']:<6.3f}  final_loss={r['final_loss']:<10.4f}  steps={r['steps']}")
