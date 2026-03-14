"""
=== MODULE 04.3: UNIVERSAL APPROXIMATION ===

THE BIG THEOREM:
  A neural network with ONE hidden layer and enough neurons
  can approximate ANY continuous function to any desired accuracy.
  — Cybenko (1989), Hornik (1991)

  This is the theoretical backbone of why neural networks work.
  It's like saying: "With enough Lego bricks, you can build anything."
  The question is just: HOW MANY bricks do you need?

HOW IT WORKS — the intuition:

  Each hidden neuron with ReLU is basically a "hinge":
    neuron(x) = max(0, w*x + b)

  A single ReLU neuron outputs a line that bends at one point.
  TWO neurons can create a "bump" (one goes up, other goes down).
  FOUR neurons can create two bumps.
  N neurons can create ~N/2 bumps.

  To approximate a wiggly function, you need enough bumps to
  match all its wiggles. More neurons → more bumps → finer fit.

THE MESH REFINEMENT ANALOGY:
  In finite element methods (FEM), you approximate a smooth shape
  with a MESH of flat pieces. Coarse mesh = rough approximation.
  Refine the mesh (more pieces) = better approximation.

  Neural networks do the same thing! Each neuron contributes a
  "basis function" (like a FEM element), and the network learns
  to weight and position these basis functions to match the target.

  1 neuron  → 1 hinge  → very rough (like 2-element mesh)
  4 neurons → 4 hinges → captures main shape
  16 neurons → 16 hinges → captures details
  64 neurons → 64 hinges → very close fit

  More neurons = more degrees of freedom = finer mesh = better fit.

WHAT THIS MODULE SHOWS:
  We train a [1, N, 1] network (1 input, N hidden, 1 output)
  to fit various 1D functions. We increase N from 1 to 64 and
  watch the approximation get better — just like refining a mesh.
"""

import numpy as np


# ── Activation functions ─────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)


# ── Target functions ─────────────────────────

def target_sine(x):
    """Simple sine wave — smooth and periodic."""
    return np.sin(2 * np.pi * x)

def target_square(x):
    """Square wave — sharp discontinuities, very hard to approximate."""
    return np.sign(np.sin(2 * np.pi * x))

def target_step(x):
    """Step function — one sharp jump at x=0.5."""
    return (x > 0.5).astype(float)

def target_wiggly(x):
    """Multi-frequency wiggly function — needs many neurons."""
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)

def target_bump(x):
    """Gaussian bump — smooth but localized."""
    return np.exp(-((x - 0.5) ** 2) / 0.02)

TARGET_FNS = {
    'sine': target_sine,
    'square': target_square,
    'step': target_step,
    'wiggly': target_wiggly,
    'bump': target_bump,
}


# ── The Network ─────────────────────────

class UniversalApproxNet:
    """
    A [1, N, 1] network trained from scratch with numpy.
    1 input → N hidden (ReLU) → 1 output (linear).

    We use a LINEAR output (no activation) because we're doing
    regression — we want to output any real value, not just 0-1.
    """

    def __init__(self, n_hidden, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_hidden = n_hidden

        # Layer 1: input(1) → hidden(N)
        # Spread the initial weights so neurons cover different parts of the input
        self.W1 = np.random.randn(n_hidden, 1) * 3.0
        self.b1 = np.linspace(-2, 2, n_hidden) + np.random.randn(n_hidden) * 0.3

        # Layer 2: hidden(N) → output(1), LINEAR output for regression
        self.W2 = np.random.randn(1, n_hidden) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros(1)

    def forward(self, x):
        """x: shape (batch,) → output: shape (batch,)"""
        X = x.reshape(-1, 1)  # (batch, 1)
        self.z1 = X @ self.W1.T + self.b1     # (batch, n_hidden)
        self.a1 = relu(self.z1)                 # (batch, n_hidden)
        self.z2 = self.a1 @ self.W2.T + self.b2  # (batch, 1)
        return self.z2.flatten()  # linear output

    def get_neuron_contributions(self, x):
        """Get each hidden neuron's individual contribution to the output."""
        X = x.reshape(-1, 1)
        z1 = X @ self.W1.T + self.b1
        a1 = relu(z1)  # (batch, n_hidden)

        # Each neuron's contribution = its activation × its output weight
        contributions = []
        for i in range(self.n_hidden):
            contrib = a1[:, i] * self.W2[0, i]  # (batch,)
            contributions.append(contrib.tolist())
        return contributions

    def train(self, x, y, epochs=2000, lr=0.01):
        """Train with SGD + momentum + MSE loss. Returns loss history."""
        losses = []
        n = len(x)
        momentum = 0.9

        # Initialize velocity terms for momentum
        v_W1 = np.zeros_like(self.W1)
        v_b1 = np.zeros_like(self.b1)
        v_W2 = np.zeros_like(self.W2)
        v_b2 = np.zeros_like(self.b2)

        for epoch in range(epochs):
            # Learning rate decay — reduce by half every 1/3 of training
            current_lr = lr * (0.5 ** (epoch // (epochs // 3 + 1)))

            # Forward
            pred = self.forward(x)

            # MSE loss
            error = pred - y
            loss = np.mean(error ** 2)
            losses.append(float(loss))

            # Backward — manual gradients
            # dL/dpred = 2 * error / n
            d_pred = 2 * error / n  # (batch,)

            # Output layer: z2 = a1 @ W2.T + b2, linear output → dL/dz2 = d_pred
            d_z2 = d_pred.reshape(-1, 1)  # (batch, 1)

            d_W2 = d_z2.T @ self.a1  # (1, n_hidden)
            d_b2 = d_z2.sum(axis=0)  # (1,)

            # Hidden layer
            d_a1 = d_z2 @ self.W2  # (batch, n_hidden)
            d_z1 = d_a1 * relu_deriv(self.z1)  # (batch, n_hidden)

            d_W1 = d_z1.T @ x.reshape(-1, 1)  # (n_hidden, 1)
            d_b1 = d_z1.sum(axis=0)  # (n_hidden,)

            # SGD with momentum update
            v_W2 = momentum * v_W2 - current_lr * d_W2
            v_b2 = momentum * v_b2 - current_lr * d_b2
            v_W1 = momentum * v_W1 - current_lr * d_W1
            v_b1 = momentum * v_b1 - current_lr * d_b1

            self.W2 += v_W2
            self.b2 += v_b2
            self.W1 += v_W1
            self.b1 += v_b1

        return losses


# ── Main API ─────────────────────────

def fit_with_increasing_neurons(function_name, neuron_counts=None, epochs=3000, seed=42):
    """
    Train networks with different hidden sizes on the same target function.

    Returns fitted curves for each neuron count, plus the target function,
    plus individual neuron contributions for the largest network.
    """
    if neuron_counts is None:
        neuron_counts = [1, 2, 4, 8, 16, 32, 64]

    fn = TARGET_FNS.get(function_name, target_sine)

    # Generate training data
    x_train = np.linspace(0, 1, 200)
    y_train = fn(x_train)

    # Evaluation points (denser for smooth curves)
    x_eval = np.linspace(0, 1, 300)
    y_target = fn(x_eval)

    # Normalize target for training stability
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    y_norm = (y_train - y_mean) / y_std

    results = []
    for n in neuron_counts:
        # Adapt learning rate based on network size
        lr = 0.02 if n <= 2 else (0.01 if n <= 8 else (0.005 if n <= 32 else 0.003))

        # Try a few seeds and keep the best — random init matters a lot
        n_tries = 3 if n <= 8 else 2
        best_net = None
        best_loss = float('inf')
        best_losses = None

        for trial in range(n_tries):
            trial_seed = seed + trial * 100
            net = UniversalApproxNet(n, seed=trial_seed)
            losses = net.train(x_train, y_norm, epochs=epochs, lr=lr)
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_net = net
                best_losses = losses

        # Evaluate with best network
        pred_norm = best_net.forward(x_eval)
        pred = pred_norm * y_std + y_mean  # denormalize

        # Get neuron contributions
        contributions = best_net.get_neuron_contributions(x_eval)
        # Scale contributions back
        contributions = [[v * y_std for v in c] for c in contributions]

        final_loss = best_loss * y_std ** 2  # denormalize loss

        results.append({
            'n_neurons': n,
            'fitted_curve': pred.tolist(),
            'final_loss': float(final_loss),
            'contributions': contributions,
        })

    return {
        'x_eval': x_eval.tolist(),
        'y_target': y_target.tolist(),
        'function_name': function_name,
        'results': results,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  UNIVERSAL APPROXIMATION DEMO")
    print("=" * 60)

    data = fit_with_increasing_neurons('sine', [1, 2, 4, 8, 16], epochs=2000)
    for r in data['results']:
        print(f"  {r['n_neurons']:2d} neurons → final MSE: {r['final_loss']:.6f}")
