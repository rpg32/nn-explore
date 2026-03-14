"""
=== MODULE 04.2: THE FORWARD PASS ===

Module 04.1 showed us what layers ARE. Now we watch data FLOW through them.

THE FORWARD PASS — step by step at each layer:

  1. Start with the input vector: x = [x1, x2]
  2. At each layer:
     a. Take the previous layer's output as input vector: h_in
     b. Multiply by the WEIGHT MATRIX: z = W @ h_in
        This is matrix multiplication!
        W has shape (num_neurons_this_layer, num_neurons_prev_layer)
        h_in has shape (num_neurons_prev_layer,)
        z has shape (num_neurons_this_layer,)
     c. Add the BIAS VECTOR: z = z + b
     d. Apply ACTIVATION function element-wise: h_out = activation(z)
  3. The output of the last layer is the network's prediction.

THE MATRIX MATH:

  Let's trace a concrete example: input [0.5, -0.3] through a 2→3→1 network.

  Layer 1 (hidden, 3 neurons, 2 inputs):
    W1 = [[w11, w12],    ← neuron 1's weights for input 1, input 2
           [w21, w22],    ← neuron 2's weights
           [w31, w32]]    ← neuron 3's weights

    z1 = W1 @ [0.5, -0.3] + b1
       = [w11*0.5 + w12*(-0.3) + b1_1,
          w21*0.5 + w22*(-0.3) + b1_2,
          w31*0.5 + w32*(-0.3) + b1_3]

    a1 = relu(z1)  ← element-wise max(0, z)

  Layer 2 (output, 1 neuron, 3 inputs):
    W2 = [[w11, w12, w13]]   ← 1 neuron receiving from 3 hidden neurons

    z2 = W2 @ a1 + b2
       = [w11*a1[0] + w12*a1[1] + w13*a1[2] + b2_1]

    a2 = sigmoid(z2)  ← final output

KEY INSIGHT:
  The forward pass is just repeated matrix multiply → add → activate.
  That's it. The entire "intelligence" of neural networks boils down to
  this simple pipeline applied layer by layer. What makes it powerful
  is that backpropagation (Module 05) adjusts the weights so this
  chain of simple operations produces useful outputs.
"""

import numpy as np


def sigmoid(z):
    """Logistic sigmoid: squashes any value to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    """ReLU: passes positive values through, zeros out negatives."""
    return np.maximum(0, z)


class ForwardPassMLP:
    """A multi-layer perceptron that exposes every intermediate computation."""

    def __init__(self, architecture, seed=42):
        """
        architecture: list of layer sizes, e.g. [2, 4, 3, 1]
          - First element is input size
          - Last element is output size
          - Everything in between is hidden layers
        """
        np.random.seed(seed)
        self.architecture = architecture
        self.weights = []
        self.biases = []

        # Initialize weights using He initialization
        # (good for ReLU: variance = 2/fan_in)
        for i in range(len(architecture) - 1):
            fan_in = architecture[i]
            fan_out = architecture[i + 1]
            # W has shape (fan_out, fan_in) so that W @ input works
            W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward_detailed(self, x):
        """
        Run forward pass and return EVERY intermediate value for visualization.

        Returns a list of layer steps. Each step has:
          - input_vector: what goes INTO this layer
          - weight_matrix: the W matrix
          - bias_vector: the b vector
          - z_vector: W @ input + b (pre-activation)
          - activation_name: 'relu' or 'sigmoid'
          - output_vector: activation(z) (post-activation)
          - matrix_products: the individual W[i,:] * input products (for visualization)
        """
        h = np.array(x, dtype=float)
        steps = []

        # Record the initial input
        steps.append({
            'step_type': 'input',
            'layer_index': 0,
            'layer_label': 'Input Layer',
            'output_vector': h.tolist(),
            'size': len(h),
        })

        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            is_output = (i == len(self.weights) - 1)

            # Step 1: Matrix multiplication — W @ h
            # Show EACH row of W multiplied by h (each neuron's dot product)
            row_products = []
            for row_idx in range(W.shape[0]):
                element_wise = (W[row_idx] * h).tolist()  # element-wise products
                dot_product = float(np.dot(W[row_idx], h))
                row_products.append({
                    'weights_row': W[row_idx].tolist(),
                    'input_vector': h.tolist(),
                    'element_products': element_wise,
                    'dot_product': dot_product,
                })

            # Step 2: z = W @ h + b
            z = W @ h + b

            # Step 3: activation
            if is_output:
                a = sigmoid(z)
                act_name = 'sigmoid'
            else:
                a = relu(z)
                act_name = 'relu'

            steps.append({
                'step_type': 'output' if is_output else 'hidden',
                'layer_index': i + 1,
                'layer_label': f'Output Layer' if is_output else f'Hidden Layer {i + 1}',
                'input_vector': h.tolist(),
                'weight_matrix': W.tolist(),
                'bias_vector': b.tolist(),
                'matmul_result': (W @ h).tolist(),  # before adding bias
                'z_vector': z.tolist(),              # after adding bias
                'activation_name': act_name,
                'output_vector': a.tolist(),
                'row_products': row_products,
                'size': len(a),
            })

            h = a

        return steps

    def get_info(self):
        """Summary info about the network."""
        total_weights = sum(w.size for w in self.weights)
        total_biases = sum(b.size for b in self.biases)
        return {
            'architecture': self.architecture,
            'num_layers': len(self.architecture),
            'total_params': total_weights + total_biases,
        }


def run_forward_pass(architecture, input_values, seed=42):
    """Main entry point: create network, run forward pass, return all data."""
    net = ForwardPassMLP(architecture, seed)
    x = np.array(input_values)
    steps = net.forward_detailed(x)
    info = net.get_info()
    return {'steps': steps, 'info': info}


if __name__ == '__main__':
    print("=" * 60)
    print("  THE FORWARD PASS — Step by Step")
    print("=" * 60)
    result = run_forward_pass([2, 3, 1], [0.5, -0.3])
    for step in result['steps']:
        print(f"\n  {step['layer_label']}:")
        if step['step_type'] == 'input':
            print(f"    Values: {step['output_vector']}")
        else:
            print(f"    Input:  {step['input_vector']}")
            print(f"    W@h:    {step['matmul_result']}")
            print(f"    +bias:  {step['z_vector']}")
            print(f"    {step['activation_name']}(): {step['output_vector']}")
