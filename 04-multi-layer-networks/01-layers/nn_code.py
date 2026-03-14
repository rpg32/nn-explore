"""
=== MODULE 04.1: LAYERS ===

In Modules 01-03, we worked with individual neurons.
Now we organize neurons into LAYERS and stack them.

THREE TYPES OF LAYERS:

  INPUT LAYER:
    Not actually neurons — just the raw data values.
    If your input is a 2D point (x1, x2), the input "layer" is 2 values.
    No weights, no activation. Just passes data forward.

  HIDDEN LAYERS:
    The layers between input and output.
    Each neuron connects to ALL neurons in the previous layer.
    This is called "fully connected" or "dense".

    Example: hidden layer with 4 neurons, previous layer has 2 values:
      neuron 1: activation(w1a*in1 + w1b*in2 + b1)
      neuron 2: activation(w2a*in1 + w2b*in2 + b2)
      neuron 3: activation(w3a*in1 + w3b*in2 + b3)
      neuron 4: activation(w4a*in1 + w4b*in2 + b4)

    That's 4 neurons × 2 inputs = 8 weights + 4 biases = 12 parameters.

  OUTPUT LAYER:
    The final layer that produces the prediction.
    For binary classification: 1 neuron with sigmoid → probability.
    For multi-class (10 digits): 10 neurons with softmax → 10 probabilities.

HOW THEY CONNECT:

  Every neuron in layer N connects to EVERY neuron in layer N-1.
  Layer N-1 has 3 neurons, layer N has 4 neurons:
    → 3 × 4 = 12 connection weights + 4 biases = 16 parameters.

  This is "fully connected" — every possible connection exists.
  (CNNs break this rule by using local connections, which is
   why they're more efficient for images.)

WHY HIDDEN LAYERS MATTER:

  A single neuron (Module 01.3) can only draw a STRAIGHT line
  as its decision boundary. No matter how you adjust the weights,
  it's always linear.

  Adding a hidden layer lets the network COMBINE multiple linear
  boundaries into curves and complex shapes. Each hidden neuron
  draws its own line, and the output layer combines them.

  More hidden neurons → more lines → finer decision boundaries.
  More hidden layers → ability to compose features hierarchically.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0, z)


class LayerDemo:
    """Interactive network for demonstrating layers."""

    def __init__(self, architecture, seed=42):
        """architecture: list of layer sizes, e.g., [2, 4, 3, 1]"""
        np.random.seed(seed)
        self.architecture = architecture
        self.weights = []
        self.biases = []
        for i in range(len(architecture) - 1):
            fan_in = architecture[i]
            self.weights.append(
                np.random.randn(architecture[i+1], fan_in) * np.sqrt(2.0 / fan_in)
            )
            self.biases.append(np.zeros(architecture[i+1]))

    def forward(self, x):
        """Forward pass returning ALL intermediate values."""
        layers_data = []
        layers_data.append({
            'type': 'input',
            'values': x.tolist(),
            'size': len(x),
        })

        h = x
        for i in range(len(self.weights)):
            # Raw weighted sum for each neuron
            z = self.weights[i] @ h + self.biases[i]

            # Activation
            if i == len(self.weights) - 1:
                a = sigmoid(z)  # output layer
                act_name = 'sigmoid'
            else:
                a = relu(z)  # hidden layers
                act_name = 'relu'

            # Per-neuron detail
            neurons = []
            for j in range(len(z)):
                w = self.weights[i][j]
                inputs = h
                products = (w * inputs).tolist()
                neurons.append({
                    'weights': w.tolist(),
                    'inputs': inputs.tolist(),
                    'products': products,
                    'weighted_sum': float(np.sum(w * inputs)),
                    'bias': float(self.biases[i][j]),
                    'z': float(z[j]),
                    'activation': float(a[j]),
                })

            layers_data.append({
                'type': 'hidden' if i < len(self.weights) - 1 else 'output',
                'values': a.tolist(),
                'size': len(a),
                'activation_fn': act_name,
                'neurons': neurons,
            })

            h = a

        return layers_data

    def get_info(self):
        total_weights = sum(w.size for w in self.weights)
        total_biases = sum(b.size for b in self.biases)
        connections = []
        for i in range(len(self.architecture) - 1):
            connections.append({
                'from_layer': i,
                'to_layer': i + 1,
                'from_size': self.architecture[i],
                'to_size': self.architecture[i + 1],
                'weights': self.architecture[i] * self.architecture[i + 1],
                'biases': self.architecture[i + 1],
            })
        return {
            'architecture': self.architecture,
            'num_layers': len(self.architecture),
            'num_hidden': len(self.architecture) - 2,
            'total_weights': total_weights,
            'total_biases': total_biases,
            'total_params': total_weights + total_biases,
            'connections': connections,
        }


def demo_forward(architecture, input_values, seed=42):
    """Run a demo forward pass with given architecture and inputs."""
    net = LayerDemo(architecture, seed)
    x = np.array(input_values)
    layers = net.forward(x)
    info = net.get_info()
    return {'layers': layers, 'info': info}


if __name__ == '__main__':
    print("=" * 55)
    print("  LAYERS DEMO")
    print("=" * 55)
    result = demo_forward([2, 4, 3, 1], [0.5, -0.3])
    for l in result['layers']:
        print(f"  {l['type']:6s} layer: {l['size']} {'neurons' if l['type'] != 'input' else 'values'} → {[f'{v:.3f}' for v in l['values']]}")
    info = result['info']
    print(f"  Total parameters: {info['total_params']} ({info['total_weights']} weights + {info['total_biases']} biases)")
