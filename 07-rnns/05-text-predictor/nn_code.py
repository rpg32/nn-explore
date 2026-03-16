"""
=== MODULE 07.5: BUILD A TEXT PREDICTOR ===

This is where everything comes together: we TRAIN a character-level
LSTM and watch it discover English patterns on its own.

THE TASK:
  Given a sequence of characters so far, predict the NEXT character.
  "th" → predict 'e' (because "the" is common)
  "and " → predict 't' (because "and the" is common)

THE MODEL:
  A character-level LSTM. Each character is one-hot encoded,
  fed into the LSTM one at a time. The LSTM's hidden state
  accumulates context, and an output layer predicts the next char.

  Input(vocab_size) → LSTM(hidden_size) → Linear(vocab_size) → Softmax

HOW IT LEARNS:
  1. Feed character 't' → LSTM predicts what comes next
  2. The actual next character is 'h' → compute cross-entropy loss
  3. Backpropagate to update weights (truncated BPTT)
  4. Repeat for every character in the training text

WHAT'S REMARKABLE:
  The model has NEVER been told about English grammar, spelling,
  or words. It discovers these patterns purely from seeing characters
  one at a time. After enough training:
    - It learns that 'q' is usually followed by 'u'
    - It learns that spaces come after words
    - It learns common words like "the", "and", "is"

  This is the same principle behind GPT, just at a tiny scale.

IMPLEMENTATION NOTES:
  - Full LSTM with forget/input/output gates, numpy from scratch
  - Truncated BPTT for training (backprop through limited steps)
  - Temperature-controlled sampling for generation
  - Simplified but functional — meant for understanding, not speed
"""

import numpy as np


# ============================================================
# VOCABULARY
# ============================================================
# Lowercase letters + space + period + comma
VOCAB = list('abcdefghijklmnopqrstuvwxyz .,')
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


# ============================================================
# TRAINING CORPUS
# ============================================================
# A few paragraphs of simple English — enough to learn basic patterns.
# ~600 chars of natural text with common words repeated.
TRAINING_TEXT = (
    "the cat sat on the mat. the dog sat on the log. "
    "a bird sang in the tree. the sun was warm and bright. "
    "the cat and the dog are friends. they play in the yard. "
    "the bird flies over the house. the wind blows through the trees. "
    "it is a good day. the sky is blue and the grass is green. "
    "the cat likes to sleep. the dog likes to run. "
    "they eat and drink and rest. the day is long and warm. "
    "at night the stars come out. the moon is big and round. "
    "the cat sleeps on the mat. the dog sleeps on the rug. "
    "all is quiet and still. the night is dark and calm."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """Stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def one_hot(idx, size):
    """Create a one-hot vector."""
    v = np.zeros(size)
    v[idx] = 1.0
    return v


def encode_text(text):
    """Convert text to list of character indices, skipping unknown chars."""
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


# ============================================================
# LSTM CLASS (from scratch, numpy only)
# ============================================================
class TextPredictor:
    """A character-level LSTM that learns to predict text.

    Architecture:
      char (one-hot, vocab_size) → LSTM cell (hidden_size) → Linear → Softmax → next char

    The LSTM cell has 4 gates, each is a mini neural network:
      - Forget gate:  f = sigmoid(W_f @ [h, x] + b_f)   — what to forget from memory
      - Input gate:   i = sigmoid(W_i @ [h, x] + b_i)   — what new info to add
      - Candidate:    g = tanh(W_g @ [h, x] + b_g)      — the new info itself
      - Output gate:  o = sigmoid(W_o @ [h, x] + b_o)   — what to output

    Cell update:  c_new = f * c_old + i * g    (forget some old, add some new)
    Hidden out:   h_new = o * tanh(c_new)      (filtered version of cell)
    """

    def __init__(self, hidden_size=64, seed=42):
        np.random.seed(seed)
        self.hidden_size = hidden_size
        self.vocab_size = VOCAB_SIZE

        # Combined input size: hidden state + character one-hot
        concat_size = hidden_size + VOCAB_SIZE

        # Xavier/Glorot initialization for stable training
        scale = np.sqrt(2.0 / (concat_size + hidden_size))

        # LSTM gate weights: each gate has weights for [h, x] concatenated input
        # Forget gate — controls what to erase from cell memory
        self.W_f = np.random.randn(hidden_size, concat_size) * scale
        self.b_f = np.ones(hidden_size) * 1.0  # Bias toward remembering (forget gate starts open)

        # Input gate — controls what new info to write
        self.W_i = np.random.randn(hidden_size, concat_size) * scale
        self.b_i = np.zeros(hidden_size)

        # Candidate gate — the actual new information
        self.W_g = np.random.randn(hidden_size, concat_size) * scale
        self.b_g = np.zeros(hidden_size)

        # Output gate — controls what to reveal from cell
        self.W_o = np.random.randn(hidden_size, concat_size) * scale
        self.b_o = np.zeros(hidden_size)

        # Output layer: hidden state → character probabilities
        out_scale = np.sqrt(2.0 / hidden_size)
        self.W_out = np.random.randn(VOCAB_SIZE, hidden_size) * out_scale
        self.b_out = np.zeros(VOCAB_SIZE)

        # Training stats
        self.epoch_losses = []
        self.total_epochs = 0

    def _lstm_step(self, x_onehot, h_prev, c_prev):
        """One LSTM time step.

        Args:
            x_onehot: one-hot character vector (vocab_size,)
            h_prev: previous hidden state (hidden_size,)
            c_prev: previous cell state (hidden_size,)

        Returns:
            h_new, c_new, cache (for backprop)
        """
        # Concatenate hidden state and input: [h_prev, x]
        concat = np.concatenate([h_prev, x_onehot])  # (hidden_size + vocab_size,)

        # Compute all four gates
        f = sigmoid(self.W_f @ concat + self.b_f)     # Forget gate
        i = sigmoid(self.W_i @ concat + self.b_i)     # Input gate
        g = np.tanh(self.W_g @ concat + self.b_g)     # Candidate values
        o = sigmoid(self.W_o @ concat + self.b_o)     # Output gate

        # Update cell state: forget old + add new
        c_new = f * c_prev + i * g

        # Compute new hidden state: filtered cell output
        h_new = o * np.tanh(c_new)

        # Cache everything for backprop
        cache = {
            'concat': concat,
            'f': f, 'i': i, 'g': g, 'o': o,
            'c_prev': c_prev, 'c_new': c_new,
            'h_prev': h_prev, 'h_new': h_new,
            'tanh_c': np.tanh(c_new),
        }

        return h_new, c_new, cache

    def _output_step(self, h):
        """Compute output probabilities from hidden state."""
        logits = self.W_out @ h + self.b_out
        probs = softmax(logits)
        return logits, probs

    def train_epoch(self, text=None, lr=0.01):
        """Train one full pass through the training text.

        Uses truncated BPTT: forward through all chars, backprop
        through limited windows for gradient stability.

        Returns:
            Average cross-entropy loss for this epoch.
        """
        if text is None:
            text = TRAINING_TEXT
        indices = encode_text(text)
        if len(indices) < 2:
            return 10.0

        # Truncation window for BPTT
        bptt_len = 25

        # Initialize hidden and cell state
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        total_loss = 0.0
        num_chars = 0

        # Process in chunks for truncated BPTT
        for start in range(0, len(indices) - 1, bptt_len):
            end = min(start + bptt_len, len(indices) - 1)
            chunk_len = end - start

            # ---- Forward pass through chunk ----
            caches = []
            outputs = []
            h_t, c_t = h.copy(), c.copy()

            for t in range(chunk_len):
                x_oh = one_hot(indices[start + t], VOCAB_SIZE)
                h_t, c_t, cache = self._lstm_step(x_oh, h_t, c_t)
                logits, probs = self._output_step(h_t)
                caches.append(cache)
                outputs.append({'logits': logits, 'probs': probs})

                # Cross-entropy loss: -log(probability of correct next char)
                target_idx = indices[start + t + 1]
                loss = -np.log(np.clip(probs[target_idx], 1e-8, 1.0))
                total_loss += loss
                num_chars += 1

            # ---- Backward pass (truncated BPTT) ----
            # Gradients for all weights
            dW_f = np.zeros_like(self.W_f)
            db_f = np.zeros_like(self.b_f)
            dW_i = np.zeros_like(self.W_i)
            db_i = np.zeros_like(self.b_i)
            dW_g = np.zeros_like(self.W_g)
            db_g = np.zeros_like(self.b_g)
            dW_o = np.zeros_like(self.W_o)
            db_o = np.zeros_like(self.b_o)
            dW_out = np.zeros_like(self.W_out)
            db_out = np.zeros_like(self.b_out)

            # Gradient flowing back through time
            dh_next = np.zeros(self.hidden_size)
            dc_next = np.zeros(self.hidden_size)

            for t in reversed(range(chunk_len)):
                cache = caches[t]
                out = outputs[t]
                target_idx = indices[start + t + 1]

                # Gradient of loss w.r.t. output (softmax + cross-entropy)
                dy = out['probs'].copy()
                dy[target_idx] -= 1.0  # d(loss)/d(logits) for softmax+CE

                # Output layer gradients
                dW_out += np.outer(dy, cache['h_new'])
                db_out += dy

                # Gradient w.r.t. hidden state (from output + from next timestep)
                dh = self.W_out.T @ dy + dh_next

                # Gradient through output gate
                do = dh * cache['tanh_c']
                do_raw = do * cache['o'] * (1 - cache['o'])  # sigmoid derivative

                # Gradient through cell state
                dc = dh * cache['o'] * (1 - cache['tanh_c']**2) + dc_next

                # Gradient through forget gate
                df = dc * cache['c_prev']
                df_raw = df * cache['f'] * (1 - cache['f'])

                # Gradient through input gate
                di = dc * cache['g']
                di_raw = di * cache['i'] * (1 - cache['i'])

                # Gradient through candidate
                dg = dc * cache['i']
                dg_raw = dg * (1 - cache['g']**2)  # tanh derivative

                # Accumulate weight gradients
                concat = cache['concat']
                dW_f += np.outer(df_raw, concat)
                db_f += df_raw
                dW_i += np.outer(di_raw, concat)
                db_i += di_raw
                dW_g += np.outer(dg_raw, concat)
                db_g += dg_raw
                dW_o += np.outer(do_raw, concat)
                db_o += do_raw

                # Gradient w.r.t. concat input (for next step back in time)
                d_concat = (self.W_f.T @ df_raw +
                            self.W_i.T @ di_raw +
                            self.W_g.T @ dg_raw +
                            self.W_o.T @ do_raw)

                # Split gradient into dh_prev and dx (we don't need dx)
                dh_next = d_concat[:self.hidden_size]
                dc_next = dc * cache['f']  # gradient flows through forget gate

            # ---- Gradient clipping (prevent exploding gradients) ----
            all_grads = [dW_f, db_f, dW_i, db_i, dW_g, db_g, dW_o, db_o, dW_out, db_out]
            total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
            max_norm = 5.0
            if total_norm > max_norm:
                scale = max_norm / total_norm
                for g in all_grads:
                    g *= scale

            # ---- Update weights ----
            self.W_f -= lr * dW_f
            self.b_f -= lr * db_f
            self.W_i -= lr * dW_i
            self.b_i -= lr * db_i
            self.W_g -= lr * dW_g
            self.b_g -= lr * db_g
            self.W_o -= lr * dW_o
            self.b_o -= lr * db_o
            self.W_out -= lr * dW_out
            self.b_out -= lr * db_out

            # Carry hidden state forward (detached — truncated BPTT)
            h, c = h_t.copy(), c_t.copy()

        avg_loss = total_loss / max(num_chars, 1)
        self.epoch_losses.append(float(avg_loss))
        self.total_epochs += 1
        return float(avg_loss)

    def generate(self, seed_text='the ', length=100, temperature=0.8):
        """Generate text character by character.

        Args:
            seed_text: starting characters to prime the LSTM
            length: how many characters to generate
            temperature: controls randomness
                - Low (0.1-0.5): confident, repetitive, safe choices
                - Medium (0.5-1.0): balanced, natural-sounding
                - High (1.0-2.0): creative, chaotic, surprising

        Returns:
            Generated text string
        """
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        # Prime with seed text (feed it through without generating)
        for ch in seed_text.lower():
            if ch in CHAR_TO_IDX:
                x = one_hot(CHAR_TO_IDX[ch], VOCAB_SIZE)
                h, c, _ = self._lstm_step(x, h, c)

        # Generate new characters
        generated = list(seed_text.lower())
        # Start generating from the last character of seed
        last_idx = CHAR_TO_IDX.get(seed_text[-1].lower(), 0) if seed_text else 0

        for _ in range(length):
            x = one_hot(last_idx, VOCAB_SIZE)
            h, c, _ = self._lstm_step(x, h, c)
            logits, _ = self._output_step(h)

            # Apply temperature: divide logits before softmax
            # Low temp → sharper distribution (confident)
            # High temp → flatter distribution (random)
            scaled_logits = logits / max(temperature, 0.01)
            probs = softmax(scaled_logits)

            # Sample from the distribution
            idx = np.random.choice(VOCAB_SIZE, p=probs)
            generated.append(IDX_TO_CHAR[idx])
            last_idx = idx

        return ''.join(generated)

    def get_predictions(self, text):
        """Get the model's top-5 predictions at each character position.

        For each character in the text, shows what the model thinks
        will come next — and whether it was right.

        Returns:
            List of dicts with predictions at each position.
        """
        indices = encode_text(text)
        if len(indices) < 1:
            return []

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        results = []

        for t in range(len(indices)):
            x = one_hot(indices[t], VOCAB_SIZE)
            h, c, _ = self._lstm_step(x, h, c)
            logits, probs = self._output_step(h)

            # Top-5 predictions
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [{'char': IDX_TO_CHAR[i],
                      'prob': float(probs[i])} for i in top5_idx]

            # What actually comes next (if not the last char)
            actual_next = None
            correct = False
            if t < len(indices) - 1:
                actual_next = IDX_TO_CHAR[indices[t + 1]]
                correct = bool(top5_idx[0] == indices[t + 1])

            results.append({
                'position': t,
                'input_char': IDX_TO_CHAR[indices[t]],
                'predictions': top5,
                'actual_next': actual_next,
                'top1_correct': correct,
            })

        return results

    def get_state(self):
        """Return model state for JSON serialization."""
        return {
            'hidden_size': self.hidden_size,
            'total_epochs': self.total_epochs,
            'epoch_losses': self.epoch_losses,
            'total_params': self._count_params(),
            'vocab': VOCAB,
            'vocab_size': VOCAB_SIZE,
            'training_text_preview': TRAINING_TEXT[:200] + '...',
            'training_text_length': len(TRAINING_TEXT),
        }

    def _count_params(self):
        """Count total trainable parameters."""
        concat_size = self.hidden_size + VOCAB_SIZE
        # 4 gates: each has (hidden_size x concat_size) weights + hidden_size biases
        lstm_params = 4 * (self.hidden_size * concat_size + self.hidden_size)
        # Output layer
        out_params = VOCAB_SIZE * self.hidden_size + VOCAB_SIZE
        return lstm_params + out_params


# ============================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================
def get_corpus_info():
    """Return info about the training corpus."""
    text = TRAINING_TEXT
    char_counts = {}
    for c in text:
        if c in CHAR_TO_IDX:
            char_counts[c] = char_counts.get(c, 0) + 1
    total = sum(char_counts.values())

    # Sort by frequency
    sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])

    return {
        'text': text,
        'length': len(text),
        'unique_chars': len(char_counts),
        'vocab_size': VOCAB_SIZE,
        'char_frequencies': [{'char': c, 'count': n, 'pct': n / total * 100}
                             for c, n in sorted_chars],
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  TEXT PREDICTOR — Character-Level LSTM")
    print("=" * 55)

    model = TextPredictor(hidden_size=64, seed=42)
    info = model.get_state()
    print(f"  Vocab: {info['vocab_size']} chars, Hidden: {info['hidden_size']}")
    print(f"  Parameters: {info['total_params']:,}")
    print(f"  Training text: {info['training_text_length']} chars")
    print()

    # Train a few epochs
    for epoch in range(5):
        loss = model.train_epoch(lr=0.01)
        print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

    # Generate some text
    print(f"\n  Generated: {model.generate('the ', length=50, temperature=0.8)}")

    # Show predictions
    preds = model.get_predictions('the cat')
    print(f"\n  Predictions for 'the cat':")
    for p in preds:
        top = p['predictions'][0]
        actual = p['actual_next'] or '(end)'
        mark = '✓' if p['top1_correct'] else '✗'
        print(f"    '{p['input_char']}' → predict '{top['char']}' ({top['prob']:.1%})  "
              f"actual: '{actual}' {mark}")
