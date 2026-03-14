"""
=== MODULE 03.1: THE CHAIN RULE ===

In Module 02.4, we computed gradients for a single neuron.
It was simple because the connection was direct:

    loss  depends on  prediction  depends on  weights

But what about DEEP networks with many layers? The loss is a
function of a function of a function...

    loss = L( sigmoid( w2 * sigmoid( w1 * x + b1 ) + b2 ) )

How does changing w1 affect the loss way at the end?
Answer: THE CHAIN RULE.

THE CHAIN RULE (the single most important idea in deep learning):

    If  y = f(g(x))  then  dy/dx = f'(g(x)) * g'(x)

    In words: the derivative through a chain of functions is
    the PRODUCT of all the local derivatives along the chain.

EXAMPLE — A Neuron's Computation Graph:

    x ──[× w]──> a ──[+ b]──> z ──[sigmoid]──> pred ──[loss]──> L

    Forward pass (left to right):
      a    = w * x
      z    = a + b
      pred = sigmoid(z)
      L    = (pred - y_true)^2

    Backward pass (right to left — the chain rule!):
      dL/dL    = 1                          (trivially)
      dL/dpred = 2 * (pred - y_true)        (derivative of squared error)
      dL/dz    = dL/dpred * pred*(1-pred)   (derivative of sigmoid)
      dL/da    = dL/dz * 1                  (derivative of addition)
      dL/dw    = dL/da * x                  (derivative of multiplication)
      dL/dx    = dL/da * w                  (derivative of multiplication)
      dL/db    = dL/dz * 1                  (derivative of addition)

    Each step just multiplies the incoming gradient by the LOCAL derivative.
    That's it. That's backpropagation.

WHY IT MATTERS:
    Without the chain rule, we'd have to compute each gradient from scratch.
    With it, we compute all gradients in a single backward sweep — efficient
    even for networks with millions of parameters.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_graph(x, w, b, y_true):
    """Run forward and backward through a single neuron's computation graph.

    Returns every intermediate value and gradient for visualization.
    """
    # ===== FORWARD PASS (left to right) =====
    a = w * x                    # multiply
    z = a + b                    # add bias
    pred = float(sigmoid(np.array(z)))  # activation
    L = (pred - y_true) ** 2     # loss (MSE for simplicity)

    # ===== BACKWARD PASS (right to left, chain rule!) =====
    # Start from the end: dL/dL = 1
    dL_dL = 1.0

    # Step back through loss: L = (pred - y_true)^2
    #   dL/dpred = 2 * (pred - y_true)
    dL_dpred = dL_dL * 2 * (pred - y_true)

    # Step back through sigmoid: pred = sigmoid(z)
    #   dpred/dz = pred * (1 - pred)
    local_sigmoid = pred * (1 - pred)
    dL_dz = dL_dpred * local_sigmoid

    # Step back through addition: z = a + b
    #   dz/da = 1,  dz/db = 1
    dL_da = dL_dz * 1.0
    dL_db = dL_dz * 1.0

    # Step back through multiplication: a = w * x
    #   da/dw = x,  da/dx = w
    dL_dw = dL_da * x
    dL_dx = dL_da * w

    return {
        # Forward values
        'forward': {
            'x': x, 'w': w, 'b': b, 'y_true': y_true,
            'a': a,        # w * x
            'z': z,        # a + b
            'pred': pred,  # sigmoid(z)
            'L': L,        # (pred - y_true)^2
        },
        # Backward gradients — each includes the local derivative
        'backward': {
            'dL_dL': dL_dL,
            'dL_dpred': dL_dpred,
            'local_loss': 2 * (pred - y_true),  # the local derivative at this node
            'dL_dz': dL_dz,
            'local_sigmoid': local_sigmoid,
            'dL_da': dL_da,
            'local_add_a': 1.0,
            'dL_db': dL_db,
            'local_add_b': 1.0,
            'dL_dw': dL_dw,
            'local_mul_w': x,  # da/dw = x
            'dL_dx': dL_dx,
            'local_mul_x': w,  # da/dx = w
        },
        # The chain shown explicitly
        'chains': {
            'dL_dw': f"dL/dL * dL/dpred * dpred/dz * dz/da * da/dw = "
                     f"1 * {2*(pred-y_true):.3f} * {local_sigmoid:.3f} * 1 * {x:.2f} = {dL_dw:.4f}",
            'dL_db': f"dL/dL * dL/dpred * dpred/dz * dz/db = "
                     f"1 * {2*(pred-y_true):.3f} * {local_sigmoid:.3f} * 1 = {dL_db:.4f}",
            'dL_dx': f"dL/dL * dL/dpred * dpred/dz * dz/da * da/dx = "
                     f"1 * {2*(pred-y_true):.3f} * {local_sigmoid:.3f} * 1 * {w:.2f} = {dL_dx:.4f}",
        },
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  THE CHAIN RULE — Gradients Through a Computation Graph")
    print("=" * 60)

    result = compute_graph(x=2.0, w=0.5, b=-0.3, y_true=1.0)
    f = result['forward']
    b = result['backward']

    print(f"\n  FORWARD (left to right):")
    print(f"    x={f['x']:.2f}  --[x{f['w']:.2f}]--> a={f['a']:.3f}  --[+{f['b']:.2f}]--> "
          f"z={f['z']:.3f}  --[sigmoid]--> pred={f['pred']:.4f}  --[loss]--> L={f['L']:.4f}")

    print(f"\n  BACKWARD (right to left):")
    print(f"    dL/dL = {b['dL_dL']:.4f}")
    print(f"    dL/dpred = {b['dL_dpred']:.4f}  (local: {b['local_loss']:.4f})")
    print(f"    dL/dz = {b['dL_dz']:.4f}  (local sigmoid': {b['local_sigmoid']:.4f})")
    print(f"    dL/dw = {b['dL_dw']:.4f}  (local: x={f['x']:.2f})")
    print(f"    dL/db = {b['dL_db']:.4f}  (local: 1)")

    print(f"\n  CHAIN for dL/dw:")
    print(f"    {result['chains']['dL_dw']}")
