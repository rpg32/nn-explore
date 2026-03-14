"""
=== MODULE 01.3: A NEURON CLASSIFIES ===

Now we put Modules 01.1 and 01.2 together for a real task.

A single neuron can CLASSIFY — it can look at a data point and decide
which of two groups it belongs to. This is called binary classification.

HOW IT WORKS:
  1. The neuron takes 2D coordinates (x1, x2) as input
  2. It computes: z = w1*x1 + w2*x2 + bias
  3. It passes z through sigmoid: output = sigmoid(z)
  4. If output >= 0.5 → Class A (blue)
     If output <  0.5 → Class B (orange)

THE DECISION BOUNDARY:
  The magic is in the decision boundary — the line where the neuron
  switches from one class to the other.

  The boundary is where: w1*x1 + w2*x2 + bias = 0

  This is a STRAIGHT LINE. Rearranging:
      x2 = -(w1/w2)*x1 - (bias/w2)

  So:
  - w1/w2 controls the SLOPE of the line
  - bias/w2 controls where the line INTERCEPTS

  The neuron can only draw straight-line boundaries. This is both
  its strength (simple, interpretable) and its limitation (can't
  handle curved or complex patterns — we need more neurons for that).

WHAT THE WEIGHTS MEAN GEOMETRICALLY:
  The weight vector [w1, w2] is PERPENDICULAR to the decision boundary.
  It points toward the "positive" (Class A) side.
  The bias shifts the line toward or away from the origin.

WHY THIS MATTERS:
  Classification is the bread and butter of neural networks:
  - Is this email spam or not?
  - Is this image a cat or a dog?
  - Is this transaction fraudulent?

  A single neuron can only handle linearly separable data (data you
  can split with a straight line). In Module 04, we'll stack neurons
  to handle any pattern.
"""

import numpy as np


def sigmoid(z):
    """Sigmoid activation — squashes values to (0, 1) for probability."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class ClassifierNeuron:
    """A single neuron that classifies 2D points.

    Takes (x1, x2) as input, outputs a probability of being Class A.
    """

    def __init__(self):
        """Start with default weights that create a simple boundary."""
        self.w1 = 1.0
        self.w2 = 1.0
        self.bias = 0.0

    def predict_single(self, x1, x2):
        """Classify a single point. Returns probability of Class A."""
        z = self.w1 * x1 + self.w2 * x2 + self.bias
        prob = float(sigmoid(np.array(z)))
        return {
            'x1': x1,
            'x2': x2,
            'weighted_sum': float(z),
            'probability': prob,
            'class': 'A' if prob >= 0.5 else 'B',
        }

    def predict_grid(self, x_min=-5, x_max=5, resolution=60):
        """Classify a whole grid of points (for the heatmap visualization).

        Returns a 2D array of probabilities that the frontend can render.
        """
        x1_vals = np.linspace(x_min, x_max, resolution)
        x2_vals = np.linspace(x_min, x_max, resolution)
        x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

        z = self.w1 * x1_grid + self.w2 * x2_grid + self.bias
        probs = sigmoid(z)

        return {
            'x1_vals': x1_vals.tolist(),
            'x2_vals': x2_vals.tolist(),
            'probabilities': probs.tolist(),
        }

    def get_decision_boundary(self, x_min=-5, x_max=5):
        """Compute the decision boundary line endpoints.

        The boundary is where w1*x1 + w2*x2 + bias = 0.

        If w2 != 0:  x2 = -(w1/w2)*x1 - (bias/w2)
        If w2 == 0:  x1 = -bias/w1  (vertical line)
        """
        if abs(self.w2) > 1e-8:
            # Normal case: solve for x2
            slope = -self.w1 / self.w2
            intercept = -self.bias / self.w2
            x1_start = x_min
            x1_end = x_max
            x2_start = slope * x1_start + intercept
            x2_end = slope * x1_end + intercept
            return {
                'type': 'line',
                'x1': [x1_start, x1_end],
                'x2': [x2_start, x2_end],
                'slope': slope,
                'intercept': intercept,
                'equation': f'x2 = {slope:.2f} * x1 + {intercept:.2f}',
            }
        elif abs(self.w1) > 1e-8:
            # Vertical line: w1*x1 + bias = 0 → x1 = -bias/w1
            x1_val = -self.bias / self.w1
            return {
                'type': 'vertical',
                'x1_val': x1_val,
                'equation': f'x1 = {x1_val:.2f}',
            }
        else:
            # Both weights zero — no boundary
            return {'type': 'none', 'equation': 'No boundary (all weights are 0)'}

    def set_params(self, w1, w2, bias):
        """Set the neuron's parameters."""
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.bias = float(bias)

    def compute_accuracy(self, points):
        """How many of the given labeled points does the neuron classify correctly?

        points: list of {'x1': float, 'x2': float, 'label': 'A' or 'B'}
        """
        if not points:
            return {'correct': 0, 'total': 0, 'accuracy': 0.0}

        correct = 0
        for p in points:
            pred = self.predict_single(p['x1'], p['x2'])
            if pred['class'] == p['label']:
                correct += 1

        return {
            'correct': correct,
            'total': len(points),
            'accuracy': correct / len(points),
        }


# ============================================================
# SAMPLE DATASETS — pre-made point sets to classify
# ============================================================

def make_dataset(name):
    """Generate a sample dataset for classification.

    Returns list of {'x1': float, 'x2': float, 'label': 'A' or 'B'}
    """
    np.random.seed(42)

    if name == 'simple':
        # Two clusters, easily separable by a horizontal-ish line
        a_points = np.random.randn(15, 2) * 0.8 + np.array([1.5, 1.5])
        b_points = np.random.randn(15, 2) * 0.8 + np.array([-1.5, -1.5])

    elif name == 'diagonal':
        # Points above and below the diagonal
        a_points = np.random.randn(15, 2) * 0.6 + np.array([2, -1])
        b_points = np.random.randn(15, 2) * 0.6 + np.array([-1, 2])

    elif name == 'close':
        # Clusters close together — harder to separate
        a_points = np.random.randn(15, 2) * 1.0 + np.array([0.8, 0.8])
        b_points = np.random.randn(15, 2) * 1.0 + np.array([-0.8, -0.8])

    elif name == 'circle':
        # Points in a ring around a center cluster — NOT linearly separable!
        # This shows the limitation of a single neuron
        angles = np.random.uniform(0, 2 * np.pi, 20)
        radii = np.random.uniform(2.5, 3.5, 20)
        a_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        b_points = np.random.randn(15, 2) * 0.6

    else:
        return []

    points = []
    for p in a_points:
        points.append({'x1': round(float(p[0]), 3), 'x2': round(float(p[1]), 3), 'label': 'A'})
    for p in b_points:
        points.append({'x1': round(float(p[0]), 3), 'x2': round(float(p[1]), 3), 'label': 'B'})

    return points


DATASETS = {
    'simple':   'Two easy clusters — a warm-up',
    'diagonal': 'Separated along the diagonal',
    'close':    'Overlapping clusters — tricky!',
    'circle':   'Ring around a cluster — impossible for one neuron!',
}


# ============================================================
# TRY IT YOURSELF
# ============================================================
if __name__ == '__main__':
    neuron = ClassifierNeuron()
    neuron.set_params(w1=1.0, w2=1.0, bias=0.0)

    print("=" * 50)
    print("  A NEURON CLASSIFIES")
    print("=" * 50)
    print()

    # Classify some test points
    test_points = [(2, 2), (-1, -1), (0, 0), (3, -2), (-2, 3)]
    for x1, x2 in test_points:
        r = neuron.predict_single(x1, x2)
        print(f"  ({x1:+.1f}, {x2:+.1f})  ->  prob={r['probability']:.3f}  class={r['class']}")

    print()
    boundary = neuron.get_decision_boundary()
    print(f"  Decision boundary: {boundary['equation']}")
    print()

    # Test on a dataset
    for name, desc in DATASETS.items():
        points = make_dataset(name)
        acc = neuron.compute_accuracy(points)
        print(f"  {name:>10}: {acc['accuracy']:.0%} accuracy  ({desc})")
