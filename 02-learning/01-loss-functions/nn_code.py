"""
=== MODULE 02.1: LOSS FUNCTIONS ===

In Module 01, our neuron produced outputs — but how do we know
if those outputs are any GOOD?

A LOSS FUNCTION measures the gap between what the neuron predicted
and what the correct answer was. The bigger the gap, the bigger
the loss. A perfect prediction has zero loss.

This is the starting point of all learning:
  1. Make a prediction
  2. Measure the loss (how wrong it was)
  3. Adjust weights to reduce the loss   ← (next modules)
  4. Repeat

TWO MAIN LOSS FUNCTIONS:

  MSE (Mean Squared Error) — for regression (predicting numbers)
  ================================================================
      L = (y_true - y_pred)^2

  Take the difference between truth and prediction, square it.
  Squaring makes all errors positive and penalizes big errors
  more than small ones (an error of 2 costs 4x more than an
  error of 1).

  Used when predicting continuous values: temperature, price, etc.


  Binary Cross-Entropy — for classification (predicting probabilities)
  ====================================================================
      L = -[ y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) ]

  Looks scary, but the intuition is simple:
    - If truth = 1 and prediction = 0.99 → loss is tiny  (0.01)
    - If truth = 1 and prediction = 0.01 → loss is HUGE  (4.6!)

  Cross-entropy SCREAMS at confident wrong predictions.
  That's exactly what we want for classification.


  WHY CROSS-ENTROPY BEATS MSE FOR CLASSIFICATION:
  =================================================
  Suppose the true label is 1 and our prediction is 0.01 (very wrong).

    MSE  loss = (1 - 0.01)^2 = 0.98     ← "eh, that's wrong"
    BCE  loss = -log(0.01)   = 4.61      ← "THAT'S VERY WRONG!"

  BCE has a much steeper gradient for wrong predictions, which
  means the network learns faster when it's making mistakes.
  MSE is "too polite" about classification errors.
"""

import numpy as np


# Prevent log(0) explosions
EPS = 1e-7


def mse_loss(y_true, y_pred):
    """Mean Squared Error: (y_true - y_pred)^2

    Works for any kind of prediction. Simple and intuitive.
    """
    return float((y_true - y_pred) ** 2)


def mse_curve(y_true, num_points=200):
    """MSE loss as a function of y_pred, for a fixed y_true."""
    y_pred = np.linspace(0, 1, num_points)
    losses = (y_true - y_pred) ** 2
    return {'y_pred': y_pred.tolist(), 'loss': losses.tolist()}


def bce_loss(y_true, y_pred):
    """Binary Cross-Entropy loss.

    The standard loss function for binary classification.
    y_pred should be between 0 and 1 (a probability).
    """
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(loss)


def bce_curve(y_true, num_points=200):
    """BCE loss as a function of y_pred, for a fixed y_true."""
    y_pred = np.linspace(EPS, 1 - EPS, num_points)
    losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return {'y_pred': y_pred.tolist(), 'loss': losses.tolist()}


def compute_both(y_true, y_pred):
    """Compute both loss functions for a given true/predicted pair."""
    return {
        'y_true': float(y_true),
        'y_pred': float(y_pred),
        'mse': mse_loss(y_true, y_pred),
        'bce': bce_loss(y_true, y_pred),
    }


def compute_batch_loss(predictions, loss_type='bce'):
    """Compute total loss over multiple predictions.

    predictions: list of {'y_true': float, 'y_pred': float}
    Returns individual losses and the mean.
    """
    loss_fn = bce_loss if loss_type == 'bce' else mse_loss
    individual = []
    for p in predictions:
        l = loss_fn(p['y_true'], p['y_pred'])
        individual.append({
            'y_true': p['y_true'],
            'y_pred': p['y_pred'],
            'loss': l,
        })
    total = sum(item['loss'] for item in individual)
    mean = total / len(individual) if individual else 0
    return {
        'individual': individual,
        'total': float(total),
        'mean': float(mean),
        'loss_type': loss_type,
    }


# A sample batch of predictions to play with
SAMPLE_BATCH = [
    {'y_true': 1.0, 'y_pred': 0.9},   # good prediction
    {'y_true': 0.0, 'y_pred': 0.1},   # good prediction
    {'y_true': 1.0, 'y_pred': 0.3},   # poor prediction
    {'y_true': 0.0, 'y_pred': 0.8},   # poor prediction
    {'y_true': 1.0, 'y_pred': 0.99},  # excellent prediction
]


# ============================================================
# TRY IT YOURSELF
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  LOSS FUNCTIONS: MEASURING HOW WRONG YOU ARE")
    print("=" * 55)
    print()

    print("  Single prediction comparisons:")
    print("  (truth=1.0)")
    print(f"  {'Predicted':>10}  {'MSE':>8}  {'BCE':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for pred in [0.01, 0.1, 0.5, 0.9, 0.99]:
        m = mse_loss(1.0, pred)
        b = bce_loss(1.0, pred)
        print(f"  {pred:>10.2f}  {m:>8.4f}  {b:>8.4f}")

    print()
    print("  Notice: BCE penalizes 0.01 (confident wrong) much")
    print("  more harshly than MSE does. That's the key insight!")
    print()

    result = compute_batch_loss(SAMPLE_BATCH, 'bce')
    print(f"  Batch loss (BCE): mean={result['mean']:.4f}")
    for item in result['individual']:
        print(f"    true={item['y_true']:.0f} pred={item['y_pred']:.2f} loss={item['loss']:.4f}")
