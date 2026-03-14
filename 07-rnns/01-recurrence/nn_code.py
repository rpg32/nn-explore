"""
=== MODULE 07: RECURRENT NEURAL NETWORKS (RNNs) ===

MLPs and CNNs process FIXED-SIZE inputs:
  - MLP: a vector of numbers
  - CNN: a fixed-size image

But many real problems have VARIABLE-LENGTH SEQUENCES:
  - Text: "hello" (5 chars) vs "goodbye" (7 chars)
  - Time series: stock prices, sensor readings
  - Audio: spoken words of different lengths

AN RNN PROCESSES ONE ELEMENT AT A TIME, keeping a "hidden state"
that carries information from previous elements:

  h[0] = zeros              (start with blank memory)
  h[1] = tanh(W_h @ h[0] + W_x @ x[1] + b)    (process first element)
  h[2] = tanh(W_h @ h[1] + W_x @ x[2] + b)    (process second, with memory of first)
  h[3] = tanh(W_h @ h[2] + W_x @ x[3] + b)    (process third, with memory of 1st & 2nd)
  ...

At each step, the SAME weights (W_h, W_x, b) are reused.
This is weight sharing across TIME, just like convolution
shares weights across SPACE.

THE HIDDEN STATE is the network's "memory" — it's a vector
of numbers that gets updated at each time step. It encodes
a summary of everything the network has seen so far.

THE PROBLEM: VANISHING GRADIENTS (again!)

  For backpropagation through time (BPTT), gradients flow backward
  through every time step. Each step multiplies by W_h.
  After 50+ steps, gradients vanish or explode — same problem
  as Module 03.3 but across time instead of layers.

  Solution: LSTM (Long Short-Term Memory) adds a "cell state"
  with gates that control what to remember and forget.
  This creates a gradient highway — gradients can flow through
  unchanged. Same idea as ReLU fixing vanishing gradients in
  deep networks.
"""

import numpy as np


# ============================================================
# SIMPLE RNN (from scratch)
# ============================================================
class SimpleRNN:
    """A basic RNN cell that processes sequences one step at a time.

    Processes characters and tries to predict the next character.
    """

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        scale_h = np.sqrt(2.0 / hidden_size)
        scale_x = np.sqrt(2.0 / input_size)

        self.hidden_size = hidden_size
        self.W_h = np.random.randn(hidden_size, hidden_size) * scale_h
        self.W_x = np.random.randn(hidden_size, input_size) * scale_x
        self.b_h = np.zeros(hidden_size)
        self.W_out = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros(output_size)

    def step(self, x, h_prev):
        """One RNN step: process input x with previous hidden state."""
        z = self.W_h @ h_prev + self.W_x @ x + self.b_h
        h = np.tanh(z)
        logits = self.W_out @ h + self.b_out
        # Softmax for output probabilities
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        return h, probs, z

    def process_sequence(self, sequence_one_hot):
        """Process a full sequence, returning all hidden states and outputs."""
        h = np.zeros(self.hidden_size)
        states = [{'h': h.tolist(), 'h_magnitude': float(np.linalg.norm(h))}]
        outputs = []

        for t, x in enumerate(sequence_one_hot):
            h, probs, z = self.step(x, h)
            states.append({
                'h': h.tolist(),
                'h_magnitude': float(np.linalg.norm(h)),
                'pre_activation': z.tolist(),
            })
            outputs.append({
                'probs': probs.tolist(),
                'predicted': int(np.argmax(probs)),
            })

        return states, outputs


# ============================================================
# CHARACTER-LEVEL PROCESSING
# ============================================================
VOCAB = list('abcdefghijklmnopqrstuvwxyz .')
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}


def text_to_onehot(text):
    """Convert text to list of one-hot vectors."""
    result = []
    for c in text.lower():
        if c in CHAR_TO_IDX:
            vec = np.zeros(len(VOCAB))
            vec[CHAR_TO_IDX[c]] = 1.0
            result.append(vec)
    return result


def process_text(text, hidden_size=16, seed=42):
    """Process text through an RNN and return all intermediate states."""
    text = text.lower()
    valid_text = ''.join(c for c in text if c in CHAR_TO_IDX)
    if not valid_text:
        valid_text = 'hello'

    one_hot = text_to_onehot(valid_text)
    rnn = SimpleRNN(len(VOCAB), hidden_size, len(VOCAB), seed=seed)
    states, outputs = rnn.process_sequence(one_hot)

    # Format for frontend
    steps = []
    for t in range(len(valid_text)):
        char = valid_text[t]
        state = states[t + 1]
        prev_state = states[t]
        output = outputs[t]

        # Top-3 predicted next chars
        probs = output['probs']
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [{'char': IDX_TO_CHAR[i], 'prob': probs[i]} for i in top3_idx]

        steps.append({
            'char': char,
            'step': t,
            'hidden': state['h'],
            'hidden_magnitude': state['h_magnitude'],
            'prev_hidden_magnitude': prev_state['h_magnitude'],
            'top3_predictions': top3,
        })

    return {
        'text': valid_text,
        'steps': steps,
        'hidden_size': hidden_size,
        'vocab_size': len(VOCAB),
        'total_params': (hidden_size * hidden_size +      # W_h
                         hidden_size * len(VOCAB) +       # W_x
                         hidden_size +                    # b_h
                         len(VOCAB) * hidden_size +       # W_out
                         len(VOCAB)),                     # b_out
    }


# ============================================================
# GRADIENT FLOW THROUGH TIME
# ============================================================
def gradient_flow_through_time(seq_length=30, hidden_size=16, seed=42):
    """Show how gradients decay across time steps (vanishing gradient).

    Simulate: how much does changing h[0] affect the output at step T?
    """
    np.random.seed(seed)
    scale = np.sqrt(2.0 / hidden_size)
    W_h = np.random.randn(hidden_size, hidden_size) * scale * 0.5  # smaller for stability

    # Compute product of Jacobians (simplified)
    # The gradient at step T w.r.t. step t is roughly: product of (diag(1-h^2) @ W_h)
    # We approximate by tracking the norm of the gradient as it flows backward

    gradients = []
    # Forward pass with dummy input
    h = np.random.randn(hidden_size) * 0.1
    h_history = [h.copy()]
    for t in range(seq_length):
        h = np.tanh(W_h @ h)
        h_history.append(h.copy())

    # Backward pass: gradient of loss w.r.t. each h[t]
    # Start with gradient = 1 at the last step
    grad = np.ones(hidden_size)
    grad_norms = [float(np.linalg.norm(grad))]

    for t in range(seq_length - 1, -1, -1):
        h_t = h_history[t + 1]
        # Gradient through tanh: multiply by (1 - h^2)
        dtanh = 1 - h_t**2
        grad = (W_h.T @ (grad * dtanh))
        grad_norms.append(float(np.linalg.norm(grad)))

    grad_norms.reverse()

    # Normalize
    max_g = max(grad_norms) if max(grad_norms) > 0 else 1
    return {
        'gradient_norms': grad_norms,
        'gradient_norms_normalized': [g / max_g for g in grad_norms],
        'seq_length': seq_length,
    }


EXAMPLE_TEXTS = {
    'hello': 'hello',
    'the cat sat': 'the cat sat.',
    'abcabc': 'abcabcabc',
    'pattern': 'aababaab',
    'longer': 'the quick brown fox jumps over the lazy dog.',
}


if __name__ == '__main__':
    print("=" * 55)
    print("  RECURRENT NEURAL NETWORKS")
    print("=" * 55)

    r = process_text('hello', hidden_size=16)
    print(f"  Text: '{r['text']}'")
    print(f"  Hidden size: {r['hidden_size']}, Params: {r['total_params']}")
    for s in r['steps']:
        top = s['top3_predictions'][0]
        print(f"  Step {s['step']}: '{s['char']}' -> predict '{top['char']}' "
              f"({top['prob']:.2%})  |h|={s['hidden_magnitude']:.3f}")
