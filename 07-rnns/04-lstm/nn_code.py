"""
=== MODULE 07.4: LSTM — LONG SHORT-TERM MEMORY ===

The basic RNN from Module 07.2 has a fatal flaw:

  GRADIENTS VANISH AFTER ~10-20 TIME STEPS.

Every step multiplies the gradient by W_h and the tanh derivative.
After 50 steps, the gradient is essentially zero — the network
can't learn long-range dependencies like:

  "The cat, which sat on the mat in the room that Jack built, WAS happy."
  (30+ words between "cat" and "was" — RNN forgets "cat" by the time it reaches "was")

THE LSTM SOLUTION: A GRADIENT HIGHWAY

  Instead of one hidden state that gets repeatedly squished by tanh,
  LSTM adds a CELL STATE — a separate memory vector that runs through
  all time steps with only element-wise operations (multiply, add).

  Think of it like a conveyor belt running through a factory:
    - Items (information) ride the belt unchanged
    - Workers at each station can ADD items to the belt (input gate)
    - Workers can REMOVE items from the belt (forget gate)
    - Workers can INSPECT items to produce output (output gate)

  Because the cell state only undergoes multiply and add (not matrix
  multiply or tanh), gradients flow through it almost unchanged.
  This is the "gradient highway" — same principle as skip connections
  in ResNets, but applied across time instead of across layers.

THE THREE GATES:

  1. FORGET GATE (f_t):  What to ERASE from the cell state
     f_t = sigmoid(W_f @ [h_prev, x_t] + b_f)
     Values near 0 = forget this memory slot
     Values near 1 = keep this memory slot

  2. INPUT GATE (i_t):   What NEW information to ADD
     i_t = sigmoid(W_i @ [h_prev, x_t] + b_i)    (how much to add)
     c̃_t = tanh(W_c @ [h_prev, x_t] + b_c)       (what to add)
     Together: i_t * c̃_t = gated new information

  3. OUTPUT GATE (o_t):  What to EXPOSE from memory
     o_t = sigmoid(W_o @ [h_prev, x_t] + b_o)
     h_t = o_t * tanh(c_t)  (selectively expose cell contents)

CELL STATE UPDATE (the key equation!):

  c_t = f_t * c_prev  +  i_t * c̃_t
        ↑ keep some      ↑ add some
        old memory        new memory

  This is just element-wise multiply and add!
  No matrix multiplication = no vanishing gradient.
  Gradients flow backward through c_t → c_{t-1} → c_{t-2} → ...
  almost unchanged, like a highway.
"""

import numpy as np


# ============================================================
# CHARACTER VOCABULARY (same as Module 07.2)
# ============================================================
VOCAB = list('abcdefghijklmnopqrstuvwxyz .')
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def text_to_onehot(text):
    """Convert text to list of one-hot vectors."""
    result = []
    for c in text.lower():
        if c in CHAR_TO_IDX:
            vec = np.zeros(VOCAB_SIZE)
            vec[CHAR_TO_IDX[c]] = 1.0
            result.append(vec)
    return result


# ============================================================
# SIGMOID — the gate activation function
# ============================================================
def sigmoid(x):
    """Sigmoid squishes values to [0, 1].

    This is perfect for gates:
      0 = fully closed (block everything)
      1 = fully open (let everything through)
    """
    # Clip for numerical stability
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# LSTM CELL — one step of the LSTM computation
# ============================================================
def lstm_step(x, h_prev, c_prev, weights):
    """One LSTM time step. This is the core of the entire module.

    Inputs:
        x:      current input vector           (input_size,)
        h_prev: previous hidden state          (hidden_size,)
        c_prev: previous cell state            (hidden_size,)
        weights: dict with all weight matrices and biases

    Returns:
        h_new:  new hidden state               (hidden_size,)
        c_new:  new cell state                  (hidden_size,)
        gates:  dict with all gate values (for visualization)

    The cell state is the gradient highway. Watch how it updates:
      c_new = forget_gate * c_prev + input_gate * candidate
    Only element-wise ops! No matrix multiply on the cell state path.
    """
    # Concatenate input and previous hidden state
    # This is the input to ALL four gates: [h_prev, x]
    combined = np.concatenate([h_prev, x])  # (hidden_size + input_size,)

    # ─── FORGET GATE ───
    # "What should we ERASE from the cell state?"
    # sigmoid output: 0 = forget completely, 1 = keep completely
    f_raw = weights['W_f'] @ combined + weights['b_f']
    f_gate = sigmoid(f_raw)

    # ─── INPUT GATE ───
    # "What NEW information should we ADD to the cell state?"
    # Two parts: (1) sigmoid decides HOW MUCH to add
    #            (2) tanh creates WHAT to add (candidate values)
    i_raw = weights['W_i'] @ combined + weights['b_i']
    i_gate = sigmoid(i_raw)

    c_raw = weights['W_c'] @ combined + weights['b_c']
    c_candidate = np.tanh(c_raw)  # candidate new memory, in [-1, 1]

    # ─── CELL STATE UPDATE ───
    # THIS IS THE KEY EQUATION:
    #   c_new = f_gate * c_prev  +  i_gate * c_candidate
    #
    # f_gate * c_prev:     keep some of the old memory
    # i_gate * c_candidate: add some new information
    #
    # Only element-wise multiply and add!
    # Gradients flow through this without repeated matrix multiplication.
    c_new = f_gate * c_prev + i_gate * c_candidate

    # ─── OUTPUT GATE ───
    # "What part of the cell state should we EXPOSE as output?"
    # The cell state is the full internal memory.
    # The hidden state is a filtered view of it.
    o_raw = weights['W_o'] @ combined + weights['b_o']
    o_gate = sigmoid(o_raw)

    h_new = o_gate * np.tanh(c_new)

    # Return everything for visualization
    gates = {
        'forget': f_gate.tolist(),      # what we're keeping (1) vs erasing (0)
        'input': i_gate.tolist(),       # how much new info to add
        'candidate': c_candidate.tolist(),  # what new info looks like
        'output': o_gate.tolist(),      # what to expose
        'cell_state': c_new.tolist(),   # the gradient highway
        'hidden_state': h_new.tolist(), # the filtered output
        # Pre-activation values for inspection
        'f_raw': f_raw.tolist(),
        'i_raw': i_raw.tolist(),
        'o_raw': o_raw.tolist(),
    }

    return h_new, c_new, gates


# ============================================================
# LSTM WEIGHT INITIALIZATION
# ============================================================
def init_lstm_weights(input_size, hidden_size, seed=42):
    """Initialize LSTM weights.

    Note the forget gate bias trick: we initialize b_f to +1.0
    so the forget gate starts near 1 (= keep everything).
    This helps the LSTM learn to remember by default.
    Without this, the cell state starts by forgetting everything,
    making it harder to learn long-range dependencies.
    """
    np.random.seed(seed)
    combined_size = hidden_size + input_size
    scale = np.sqrt(2.0 / combined_size)

    weights = {
        # Forget gate weights and bias
        'W_f': np.random.randn(hidden_size, combined_size) * scale,
        'b_f': np.ones(hidden_size),  # Bias = 1 → start by remembering!

        # Input gate weights and bias
        'W_i': np.random.randn(hidden_size, combined_size) * scale,
        'b_i': np.zeros(hidden_size),

        # Candidate (cell update) weights and bias
        'W_c': np.random.randn(hidden_size, combined_size) * scale,
        'b_c': np.zeros(hidden_size),

        # Output gate weights and bias
        'W_o': np.random.randn(hidden_size, combined_size) * scale,
        'b_o': np.zeros(hidden_size),

        # Output projection (hidden state → character prediction)
        'W_out': np.random.randn(VOCAB_SIZE, hidden_size) * np.sqrt(2.0 / hidden_size),
        'b_out': np.zeros(VOCAB_SIZE),
    }
    return weights


# ============================================================
# PROCESS A FULL SEQUENCE
# ============================================================
def process_sequence(text, hidden_size=16):
    """Process text through an LSTM, returning all states and gate values.

    This is the main function called by the frontend.
    """
    text = text.lower()
    valid_text = ''.join(c for c in text if c in CHAR_TO_IDX)
    if not valid_text:
        valid_text = 'hello'

    one_hot = text_to_onehot(valid_text)
    weights = init_lstm_weights(VOCAB_SIZE, hidden_size)

    # Initial states: all zeros
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)

    steps = []
    for t, x in enumerate(one_hot):
        h_prev = h.copy()
        c_prev = c.copy()

        h, c, gates = lstm_step(x, h_prev, c_prev, weights)

        # Predict next character
        logits = weights['W_out'] @ h + weights['b_out']
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [{'char': IDX_TO_CHAR[i], 'prob': float(probs[i])} for i in top3_idx]

        steps.append({
            'char': valid_text[t],
            'step': t,
            'gates': gates,
            'prev_hidden': h_prev.tolist(),
            'prev_cell': c_prev.tolist(),
            'hidden_magnitude': float(np.linalg.norm(h)),
            'cell_magnitude': float(np.linalg.norm(c)),
            'top3_predictions': top3,
        })

    # Count parameters
    param_count = (
        4 * hidden_size * (hidden_size + VOCAB_SIZE) +  # 4 weight matrices
        4 * hidden_size +                                # 4 biases
        VOCAB_SIZE * hidden_size +                       # W_out
        VOCAB_SIZE                                       # b_out
    )

    return {
        'text': valid_text,
        'steps': steps,
        'hidden_size': hidden_size,
        'vocab_size': VOCAB_SIZE,
        'total_params': param_count,
    }


# ============================================================
# GRADIENT FLOW COMPARISON: RNN vs LSTM
# ============================================================
def compare_gradient_flow(seq_length=40, hidden_size=16, seed=42):
    """Compare how gradients decay in a basic RNN vs an LSTM.

    For the RNN: gradient at step t ≈ product of (diag(1-h²) @ W_h)
    across all steps from t to T. This decays exponentially.

    For the LSTM: gradient flows through the cell state path,
    which only involves element-wise operations. The forget gate
    controls decay, and with bias=1, most gates start near 1,
    so the gradient stays much more stable.

    We simulate both and return normalized gradient norms.
    """
    np.random.seed(seed)

    # ─── RNN gradient flow ───
    # Each step multiplies gradient by (1-h²) * W_h
    scale = np.sqrt(2.0 / hidden_size)
    W_h_rnn = np.random.randn(hidden_size, hidden_size) * scale * 0.5

    # Forward pass
    h = np.random.randn(hidden_size) * 0.1
    h_history = [h.copy()]
    for t in range(seq_length):
        h = np.tanh(W_h_rnn @ h)
        h_history.append(h.copy())

    # Backward pass
    grad_rnn = np.ones(hidden_size)
    rnn_norms = [float(np.linalg.norm(grad_rnn))]
    for t in range(seq_length - 1, -1, -1):
        dtanh = 1 - h_history[t + 1] ** 2
        grad_rnn = W_h_rnn.T @ (grad_rnn * dtanh)
        rnn_norms.append(float(np.linalg.norm(grad_rnn)))
    rnn_norms.reverse()

    # ─── LSTM gradient flow ───
    # Gradient through cell state: dc_t/dc_{t-1} = f_gate
    # With forget gate bias = 1, f_gate ≈ 0.7-0.9
    # So gradient decays MUCH more slowly

    weights = init_lstm_weights(hidden_size, hidden_size, seed=seed)
    # Use a higher forget bias for the comparison — this matches what
    # trained LSTMs learn to do (keep the forget gate near 1 for
    # important memory slots). Some implementations use bias=1 to 5.
    weights['b_f'] = np.ones(hidden_size) * 2.0

    # We simulate with dummy inputs
    h_lstm = np.zeros(hidden_size)
    c_lstm = np.zeros(hidden_size)
    forget_gates = []

    for t in range(seq_length):
        x = np.random.randn(hidden_size) * 0.3
        combined = np.concatenate([h_lstm, x])

        f_gate = sigmoid(weights['W_f'] @ combined + weights['b_f'])
        i_gate = sigmoid(weights['W_i'] @ combined + weights['b_i'])
        c_cand = np.tanh(weights['W_c'] @ combined + weights['b_c'])
        c_lstm = f_gate * c_lstm + i_gate * c_cand

        o_gate = sigmoid(weights['W_o'] @ combined + weights['b_o'])
        h_lstm = o_gate * np.tanh(c_lstm)

        forget_gates.append(f_gate.copy())

    # LSTM backward through cell state path
    # dc_t / dc_{t-1} = f_t  (just the forget gate!)
    grad_lstm = np.ones(hidden_size)
    lstm_norms = [float(np.linalg.norm(grad_lstm))]
    for t in range(seq_length - 1, -1, -1):
        grad_lstm = grad_lstm * forget_gates[t]
        lstm_norms.append(float(np.linalg.norm(grad_lstm)))
    lstm_norms.reverse()

    # Normalize both to [0, 1] using the maximum across both
    max_norm = max(max(rnn_norms), max(lstm_norms), 1e-10)
    rnn_normalized = [g / max_norm for g in rnn_norms]
    lstm_normalized = [g / max_norm for g in lstm_norms]

    return {
        'rnn_gradients': rnn_normalized,
        'lstm_gradients': lstm_normalized,
        'rnn_raw': rnn_norms,
        'lstm_raw': lstm_norms,
        'seq_length': seq_length,
        'rnn_final': rnn_normalized[0],
        'lstm_final': lstm_normalized[0],
        'avg_forget_gate': float(np.mean([f.mean() for f in forget_gates])),
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  LSTM — LONG SHORT-TERM MEMORY")
    print("=" * 60)
    print()

    # Process a sequence
    result = process_sequence('hello world', hidden_size=16)
    print(f"  Text: '{result['text']}'")
    print(f"  Hidden size: {result['hidden_size']}, Params: {result['total_params']}")
    print()
    for s in result['steps']:
        fg = np.mean(s['gates']['forget'])
        ig = np.mean(s['gates']['input'])
        og = np.mean(s['gates']['output'])
        print(f"  Step {s['step']:2d}: '{s['char']}'  "
              f"forget={fg:.2f}  input={ig:.2f}  output={og:.2f}  "
              f"|c|={s['cell_magnitude']:.3f}  |h|={s['hidden_magnitude']:.3f}")

    print()
    # Gradient comparison
    comp = compare_gradient_flow(seq_length=40)
    print(f"  Gradient after 40 steps:")
    print(f"    RNN:  {comp['rnn_final']:.6f}  (nearly zero!)")
    print(f"    LSTM: {comp['lstm_final']:.6f}  (much stronger)")
    print(f"    Average forget gate: {comp['avg_forget_gate']:.3f}")
