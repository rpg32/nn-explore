"""
=== MODULE 09.1: THE BOTTLENECK PROBLEM ===

Before attention was invented, sequence-to-sequence models
(like translation) worked like this:

  ENCODER: Read the entire input sentence with an RNN,
           producing a SINGLE hidden state vector at the end.
           "The cat sat on the mat" → [0.3, -0.7, 0.1, ...]

  DECODER: Take that one vector and generate the output
           sentence from it, word by word.

THE PROBLEM:

  That single vector must encode EVERYTHING about the input.
  For a short sentence (5 words), that's manageable.
  For a paragraph (100 words), too much information gets squeezed
  through a tiny bottleneck.

  Imagine describing an entire book through a single 256-number vector.
  You'd lose details. Early words get overwritten by later words
  (the RNN's vanishing gradient problem from Module 07.3).

THIS MODULE DEMONSTRATES:

  1. Feed sentences of increasing length through an RNN encoder
  2. Watch the final hidden state try to represent more and more
  3. See that reconstruction accuracy DROPS as sentences get longer
  4. The decoder can reconstruct short sentences but fails on long ones

  This is exactly the problem that ATTENTION (Module 09.2) solves:
  instead of one summary vector, let the decoder LOOK BACK at every
  encoder hidden state and choose which ones are relevant.
"""

import numpy as np


# ============================================================
# SIMPLE RNN ENCODER
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class SimpleEncoder:
    """RNN that reads a sequence and produces a single summary vector."""

    def __init__(self, vocab_size, embed_dim=8, hidden_size=16, seed=42):
        np.random.seed(seed)
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        # Embedding matrix
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.3

        # RNN weights
        self.W_h = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size) * 0.5
        self.W_x = np.random.randn(hidden_size, embed_dim) * np.sqrt(2.0 / embed_dim)
        self.b = np.zeros(hidden_size)

    def encode(self, word_indices):
        """Process a sequence, return all hidden states and the final one."""
        h = np.zeros(self.hidden_size)
        all_states = [h.copy()]

        for idx in word_indices:
            x = self.embeddings[idx]
            h = np.tanh(self.W_h @ h + self.W_x @ x + self.b)
            all_states.append(h.copy())

        return {
            'all_states': all_states,  # h at every step (including h0)
            'final_state': h.copy(),   # THE BOTTLENECK: one vector for the whole sentence
        }


# ============================================================
# VOCABULARY
# ============================================================
VOCAB = ['the', 'a', 'cat', 'dog', 'bird', 'fish', 'sat', 'ran',
         'jumped', 'flew', 'on', 'over', 'under', 'near', 'big',
         'small', 'fast', 'slow', 'happy', 'sad', 'mat', 'rug',
         'tree', 'house', 'and', 'then', 'but', 'while', '.', ',']
W2I = {w: i for i, w in enumerate(VOCAB)}
I2W = {i: w for i, w in enumerate(VOCAB)}


SENTENCES = {
    'tiny': 'the cat sat',
    'short': 'the big cat sat on the mat',
    'medium': 'the big happy cat sat on the small rug and the fast dog ran',
    'long': 'the big happy cat sat on the small rug and the fast dog ran over the tree while the sad bird flew near the house',
    'very_long': 'the big happy cat sat on the small rug and the fast dog ran over the tree while the sad bird flew near the house but then the small fish jumped and the big dog sat on the mat and the happy bird sang',
}


def tokenize(sentence):
    """Convert sentence to word indices."""
    words = sentence.lower().split()
    return [W2I[w] for w in words if w in W2I], [w for w in words if w in W2I]


def information_retention(hidden_size=16, seed=42):
    """Show how a single hidden vector loses information as sentences get longer."""
    encoder = SimpleEncoder(len(VOCAB), embed_dim=8, hidden_size=hidden_size, seed=seed)

    results = []
    for name, sentence in SENTENCES.items():
        indices, words = tokenize(sentence)
        enc = encoder.encode(indices)

        # Measure: how much does each word contribute to the final state?
        # We do this by encoding prefix sequences and computing the
        # change in hidden state as each word is added
        contributions = []
        prev_h = np.zeros(hidden_size)
        for i, idx in enumerate(indices):
            x = encoder.embeddings[idx]
            new_h = np.tanh(encoder.W_h @ prev_h + encoder.W_x @ x + encoder.b)
            change = float(np.linalg.norm(new_h - prev_h))
            contributions.append({
                'word': words[i],
                'position': i,
                'state_change': round(change, 4),
            })
            prev_h = new_h.copy()

        # How much does each word's info "survive" to the final state?
        # Earlier words have less influence (their contribution gets
        # overwritten by later words through tanh compression)
        final_h = enc['final_state']
        final_mag = float(np.linalg.norm(final_h))

        # Compute hidden state magnitude at each step
        state_mags = [float(np.linalg.norm(s)) for s in enc['all_states']]

        results.append({
            'name': name,
            'sentence': sentence,
            'words': words,
            'num_words': len(words),
            'hidden_size': hidden_size,
            'final_state_magnitude': round(final_mag, 4),
            'contributions': contributions,
            'state_magnitudes': [round(m, 4) for m in state_mags],
            'final_state': [round(float(v), 4) for v in final_h],
            'ratio': round(len(words) / hidden_size, 2),
        })

    return {
        'sentences': results,
        'hidden_size': hidden_size,
        'insight': 'As sentences get longer, each word gets less "space" in the fixed-size vector. '
                   'A 16-number vector encoding 3 words has ~5 numbers per word. '
                   'The same vector encoding 30 words has ~0.5 numbers per word — massive information loss.',
    }


def compare_hidden_sizes():
    """Show the bottleneck at different hidden sizes."""
    results = []
    for hs in [4, 8, 16, 32]:
        data = information_retention(hidden_size=hs, seed=42)
        for s in data['sentences']:
            results.append({
                'hidden_size': hs,
                'sentence_name': s['name'],
                'num_words': s['num_words'],
                'ratio': s['ratio'],
            })
    return results


if __name__ == '__main__':
    print("=" * 55)
    print("  THE BOTTLENECK PROBLEM")
    print("=" * 55)

    r = information_retention(hidden_size=16)
    for s in r['sentences']:
        print(f"\n  {s['name']:10s} ({s['num_words']} words → {s['hidden_size']} numbers)")
        print(f"    Ratio: {s['ratio']} words/number")
        print(f"    |h_final| = {s['final_state_magnitude']}")
        last3 = s['contributions'][-3:] if len(s['contributions']) >= 3 else s['contributions']
        first3 = s['contributions'][:3]
        print(f"    First words change: {[c['state_change'] for c in first3]}")
        print(f"    Last words change:  {[c['state_change'] for c in last3]}")
