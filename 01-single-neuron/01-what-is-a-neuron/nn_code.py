"""
=== MODULE 01.1: WHAT IS A NEURON? ===

The simplest building block of a neural network.

A neuron does one thing: it takes numbers in, and puts one number out.

    output = (w1 * x1) + (w2 * x2) + (w3 * x3) + bias

That's it. Seriously. Everything in deep learning — image recognition,
language models, ChatGPT — is built from millions of these simple units.

HOW IT WORKS:
  1. INPUTS (x1, x2, x3): Numbers fed into the neuron
  2. WEIGHTS (w1, w2, w3): How important each input is
  3. WEIGHTED SUM: Multiply each input by its weight, add them up
  4. BIAS: A constant added at the end (shifts the output up or down)
  5. OUTPUT: The final number

ANALOGY — Deciding where to eat lunch:
  - Input 1: How good is the food? (x1 = 8/10)
  - Input 2: How far away is it?   (x2 = 3/10)
  - Input 3: How cheap is it?      (x3 = 6/10)

  Your brain assigns importance (weights):
  - Food quality matters a lot:  w1 =  0.5
  - Distance matters (negative): w2 = -0.3  (farther = worse)
  - Price matters a bit:         w3 =  0.2

  Score = (0.5 × 8) + (-0.3 × 3) + (0.2 × 6) + 0.0
        = 4.0 + (-0.9) + 1.2
        = 4.3
"""

import numpy as np


class Neuron:
    """A single artificial neuron.

    This is the fundamental unit of every neural network.
    No matter how complex the network, it's made of these.
    """

    def __init__(self, num_inputs):
        """Create a neuron with random starting weights.

        Args:
            num_inputs: How many input values this neuron accepts
        """
        # Weights start random and small
        # In real networks, these get adjusted during training
        self.weights = np.random.randn(num_inputs) * 0.5

        # Bias starts at zero
        self.bias = 0.0

    def forward(self, inputs):
        """Compute the neuron's output.

        This is the core computation:
            output = sum(weight_i * input_i) + bias

        Args:
            inputs: list or numpy array of input values

        Returns:
            dict with every step of the computation (for visualization)
        """
        inputs = np.array(inputs, dtype=float)

        # Step 1: Multiply each input by its weight
        weighted_inputs = inputs * self.weights

        # Step 2: Sum all the weighted inputs
        weighted_sum = np.sum(weighted_inputs)

        # Step 3: Add the bias
        output = weighted_sum + self.bias

        # Return everything so the visualization can show each step
        return {
            'inputs': inputs.tolist(),
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'weighted_inputs': weighted_inputs.tolist(),
            'weighted_sum': float(weighted_sum),
            'output': float(output)
        }

    def set_weights(self, weights):
        """Set the neuron's weights manually."""
        self.weights = np.array(weights, dtype=float)

    def set_bias(self, bias):
        """Set the neuron's bias manually."""
        self.bias = float(bias)


# ============================================================
# TRY IT YOURSELF — Run this file directly:  python nn_code.py
# ============================================================
if __name__ == '__main__':
    # Create a neuron with 3 inputs
    neuron = Neuron(num_inputs=3)

    # Set the "restaurant" weights from the analogy above
    neuron.set_weights([0.5, -0.3, 0.2])
    neuron.set_bias(0.0)

    # Our restaurant ratings
    inputs = [8.0, 3.0, 6.0]

    # Compute!
    result = neuron.forward(inputs)

    print("=" * 45)
    print("  A NEURON DECIDING WHERE TO EAT LUNCH")
    print("=" * 45)
    print(f"  Inputs:  {result['inputs']}")
    print(f"  Weights: {result['weights']}")
    print(f"  Bias:    {result['bias']}")
    print()
    print("  Step by step:")
    labels = ['food quality', 'distance', 'price']
    for i, (w, x, wx) in enumerate(zip(
        result['weights'], result['inputs'], result['weighted_inputs']
    )):
        print(f"    w{i+1} x x{i+1} = {w:+.1f} x {x:.1f} = {wx:+.2f}  ({labels[i]})")
    print(f"    {'-' * 35}")
    print(f"    Weighted sum = {result['weighted_sum']:+.2f}")
    print(f"    + Bias       = {result['bias']:+.2f}")
    print(f"    {'-' * 35}")
    print(f"    Output       = {result['output']:+.2f}")
    print()
    print("  >> Restaurant score: {:.2f}".format(result['output']))
    print()

    # Now try different restaurants!
    print("  Let's compare another restaurant:")
    print("  (Great food=9, but far=8, and expensive=2)")
    result2 = neuron.forward([9.0, 8.0, 2.0])
    print(f"  Score: {result2['output']:.2f}")
    print()
    print("  Try editing this file to change the inputs and weights!")
