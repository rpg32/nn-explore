"""
=== MODULE 01.2: ACTIVATION FUNCTIONS ===

In Module 01.1 we learned that a neuron computes:

    weighted_sum = (w1 * x1) + (w2 * x2) + (w3 * x3) + bias

But there's a problem. That weighted sum is LINEAR — it's just
a straight-line relationship. If you stack 100 neurons that each
compute a linear function, the whole thing is STILL just linear.

    linear(linear(linear(x))) = still linear!

That means a network of linear neurons can only learn straight-line
patterns. It could never learn to recognize a face, understand
language, or do anything interesting.

THE FIX: Activation functions.

After computing the weighted sum, we pass it through a nonlinear
function — an "activation function" — that bends the output:

    output = activation(weighted_sum)

This one change is what gives neural networks their power.

COMMON ACTIVATION FUNCTIONS:

  1. STEP      — The original. Output is 0 or 1. Like a light switch.
                 If input >= 0: output = 1, else: output = 0

  2. SIGMOID   — A smooth S-curve. Squashes any number into (0, 1).
                 sigmoid(x) = 1 / (1 + e^(-x))
                 Good for probabilities. Was the default for decades.

  3. TANH      — Like sigmoid but outputs (-1, 1). Centered at zero.
                 tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                 Better than sigmoid because outputs are centered.

  4. RELU      — The modern workhorse. Dead simple.
                 relu(x) = max(0, x)
                 If positive, pass through. If negative, output 0.
                 Fast to compute, works great in practice.

WHY DOES THIS MATTER?
  With activation functions, each neuron can learn a "bend" in the
  function it represents. Stack enough neurons with bends, and you
  can approximate ANY function — curves, spirals, anything.
"""

import numpy as np


# ============================================================
# THE ACTIVATION FUNCTIONS
# ============================================================

def step(x):
    """Step function: outputs 0 or 1.

    The simplest possible activation. Like a light switch.
    Used in the original 1958 Perceptron.
    """
    return np.where(x >= 0, 1.0, 0.0)


def sigmoid(x):
    """Sigmoid: squashes any value into the range (0, 1).

    The classic activation function, shaped like an S-curve.
    Used for decades. Still used for output layers when you
    want a probability (0 to 1).

    Problem: gradients become very small for large |x| values
    (the "vanishing gradient" problem — more on this later).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    """Tanh: squashes any value into the range (-1, 1).

    Like sigmoid, but centered at zero. This matters because
    it means the average output is around 0, which helps
    learning be more stable.
    """
    return np.tanh(x)


def relu(x):
    """ReLU (Rectified Linear Unit): max(0, x).

    The most popular activation function in modern deep learning.
    - If input is positive, pass it through unchanged
    - If input is negative, output 0

    Why it works so well:
    - Dead simple and fast to compute
    - Doesn't squash large values (no vanishing gradient for positives)
    - Creates sparse activations (many neurons output exactly 0)
    """
    return np.maximum(0, x)


# Registry of all activations for the app to use
ACTIVATIONS = {
    'step':    {'fn': step,    'label': 'Step',    'color': '#ef5350', 'description': 'Binary on/off. The original 1958 activation.'},
    'sigmoid': {'fn': sigmoid, 'label': 'Sigmoid', 'color': '#4fc3f7', 'description': 'Smooth S-curve. Squashes to (0, 1). Classic choice.'},
    'tanh':    {'fn': tanh,    'label': 'Tanh',    'color': '#66bb6a', 'description': 'Like sigmoid but centered. Outputs (-1, 1).'},
    'relu':    {'fn': relu,    'label': 'ReLU',    'color': '#ffb74d', 'description': 'max(0, x). The modern default. Simple and effective.'},
}


def compute_activation(name, x):
    """Apply a named activation function to a value.

    Args:
        name: one of 'step', 'sigmoid', 'tanh', 'relu'
        x: the input value (typically the weighted sum from a neuron)

    Returns:
        dict with input, output, and function details
    """
    fn = ACTIVATIONS[name]['fn']
    output = float(fn(np.array(x)))
    return {
        'name': name,
        'input': float(x),
        'output': output,
        'label': ACTIVATIONS[name]['label'],
        'description': ACTIVATIONS[name]['description'],
    }


def compute_curve(name, x_min=-6.0, x_max=6.0, num_points=200):
    """Generate the full curve for an activation function.

    Returns arrays of x and y values for plotting.
    """
    fn = ACTIVATIONS[name]['fn']
    x = np.linspace(x_min, x_max, num_points)
    y = fn(x)
    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'color': ACTIVATIONS[name]['color'],
    }


def demonstrate_linearity_problem():
    """Show why stacking linear functions is still linear.

    linear1(x) = 2x + 1
    linear2(x) = 3x - 2
    linear2(linear1(x)) = 3(2x + 1) - 2 = 6x + 1  <-- still linear!

    But:
    relu(linear1(x)) then linear2 = something curved!
    """
    x = np.linspace(-2, 2, 100)

    # Stacking two linear functions
    linear1 = 2 * x + 1
    linear2_of_linear1 = 3 * linear1 - 2  # = 6x + 1, still a line!

    # Using ReLU between them
    activated = relu(linear1)
    linear2_of_activated = 3 * activated - 2  # NOT a line anymore!

    return {
        'x': x.tolist(),
        'linear_stacked': linear2_of_linear1.tolist(),
        'with_activation': linear2_of_activated.tolist(),
    }


# ============================================================
# TRY IT YOURSELF
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  ACTIVATION FUNCTIONS")
    print("=" * 50)
    print()

    test_values = [-3.0, -1.0, 0.0, 0.5, 2.0, 5.0]

    for name in ['step', 'sigmoid', 'tanh', 'relu']:
        info = ACTIVATIONS[name]
        print(f"  {info['label']:>8s}: ", end='')
        for x in test_values:
            y = float(info['fn'](np.array(x)))
            print(f"  f({x:+.1f})={y:+.2f}", end='')
        print()

    print()
    print("  THE LINEARITY PROBLEM:")
    print("  linear(linear(x)) = 6x + 1  (still a straight line!)")
    print("  linear(relu(linear(x)))      (now it bends!)")
    print()
    print("  Run 'python app.py' to see this interactively!")
