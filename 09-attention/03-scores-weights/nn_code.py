"""
=== MODULE 09.3: ATTENTION SCORES & WEIGHTS ===

Module 09.2 showed attention for ONE query against multiple keys.
Now we compute attention for EVERY word as a query against EVERY
word as a key. This produces a full MATRIX of attention weights.

THE ATTENTION MATRIX:

  For a sentence of N words, we get an N×N matrix where:
    matrix[i][j] = "how much does word i attend to word j"

  Each ROW sums to 1 (it's a probability distribution).
  Row i tells you: "when processing word i, how much does it
  look at each other word?"

THE COMPUTATION (for all words at once):

  1. Q = X @ W_Q    →  every word gets a query vector (N × d_k)
  2. K = X @ W_K    →  every word gets a key vector   (N × d_k)
  3. V = X @ W_V    →  every word gets a value vector  (N × d_k)

  4. scores = Q @ K^T / sqrt(d_k)   →  N×N matrix of raw scores
  5. weights = softmax(scores, per row)  →  N×N matrix of attention weights
  6. output = weights @ V            →  N × d_k context-aware embeddings

  Steps 4-6 are just THREE MATRIX MULTIPLIES. That's it.
  No loops, no recurrence. All words processed in parallel.
  This is why transformers are so fast on GPUs.

WHAT THE MATRIX SHOWS:

  - Diagonal (word attending to itself): often high
  - Off-diagonal: which OTHER words each word cares about
  - Patterns emerge: "the" attends to its noun, verbs attend
    to their subjects, adjectives attend to what they modify
"""

import numpy as np


def softmax_rows(z):
    """Softmax along each row independently."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ============================================================
# WORD EMBEDDINGS (biased by category for clear results)
# ============================================================
CATEGORIES = {
    'the': 'func', 'a': 'func', 'an': 'func', 'on': 'func',
    'in': 'func', 'to': 'func', 'and': 'func', 'is': 'func',
    'cat': 'animal', 'dog': 'animal', 'bird': 'animal', 'fish': 'animal',
    'sat': 'verb', 'ran': 'verb', 'jumped': 'verb', 'flew': 'verb',
    'chased': 'verb', 'slept': 'verb', 'ate': 'verb',
    'big': 'adj', 'small': 'adj', 'fast': 'adj', 'happy': 'adj',
    'red': 'adj', 'lazy': 'adj', 'quick': 'adj', 'brown': 'adj',
    'mat': 'noun', 'rug': 'noun', 'tree': 'noun', 'house': 'noun',
    'fox': 'animal', 'over': 'func',
}

CAT_COLORS = {
    'animal': '#4fc3f7', 'verb': '#ef5350', 'adj': '#66bb6a',
    'func': '#888888', 'noun': '#ffb74d', 'unknown': '#555555',
}

CAT_BASES = {
    'animal': np.array([1.0, 0.5, -0.3, 0.8, 0.2, -0.1]),
    'verb':   np.array([-0.5, 1.0, 0.7, -0.2, 0.3, 0.6]),
    'func':   np.array([0.1, -0.1, 0.1, -0.1, 0.05, -0.05]),
    'adj':    np.array([0.3, -0.8, 1.0, 0.4, -0.5, 0.2]),
    'noun':   np.array([0.7, 0.3, -0.5, 0.9, 0.4, -0.3]),
    'unknown': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


def get_embedding(word, embed_dim=6, seed_offset=0):
    cat = CATEGORIES.get(word.lower(), 'unknown')
    base = CAT_BASES.get(cat, CAT_BASES['unknown'])[:embed_dim]
    np.random.seed(hash(word) % 10000 + seed_offset)
    noise = np.random.randn(embed_dim) * 0.12
    return base + noise


def compute_attention_matrix(sentence, embed_dim=6, d_k=4, seed=42):
    """Compute the full NxN attention matrix for a sentence."""
    np.random.seed(seed)
    words = sentence.lower().split()
    N = len(words)

    # Build embedding matrix
    X = np.array([get_embedding(w, embed_dim) for w in words])  # (N, embed_dim)

    # Projection matrices (would be learned)
    W_Q = np.random.randn(embed_dim, d_k) * np.sqrt(2.0 / embed_dim)
    W_K = np.random.randn(embed_dim, d_k) * np.sqrt(2.0 / embed_dim)
    W_V = np.random.randn(embed_dim, d_k) * np.sqrt(2.0 / embed_dim)

    # Step 1-3: Project all words at once
    Q = X @ W_Q  # (N, d_k)
    K = X @ W_K  # (N, d_k)
    V = X @ W_V  # (N, d_k)

    # Step 4: Raw scores
    raw_scores = Q @ K.T / np.sqrt(d_k)  # (N, N)

    # Step 5: Softmax per row
    attention_weights = softmax_rows(raw_scores)  # (N, N)

    # Step 6: Weighted sum of values
    output = attention_weights @ V  # (N, d_k)

    # Format results
    categories = [CATEGORIES.get(w, 'unknown') for w in words]
    colors = [CAT_COLORS.get(c, '#555') for c in categories]

    return {
        'words': words,
        'num_words': N,
        'categories': categories,
        'colors': colors,
        'embed_dim': embed_dim,
        'd_k': d_k,

        'raw_scores': [[round(float(v), 3) for v in row] for row in raw_scores],
        'attention_weights': [[round(float(v), 3) for v in row] for row in attention_weights],

        # Per-word: who does each word attend to most?
        'top_attention': [
            {
                'word': words[i],
                'attends_to': [
                    {'word': words[j], 'weight': round(float(attention_weights[i][j]), 3)}
                    for j in np.argsort(attention_weights[i])[::-1][:3]
                ]
            }
            for i in range(N)
        ],

        'computation': {
            'Q_shape': [N, d_k],
            'K_shape': [N, d_k],
            'scores_shape': [N, N],
            'total_params': 3 * embed_dim * d_k,
        },
    }


EXAMPLE_SENTENCES = [
    'the cat sat on the mat',
    'the big dog chased the small cat',
    'a quick brown fox jumped over the lazy dog',
    'the happy bird flew and the sad fish sat',
    'the cat and the dog sat on the big red mat',
]


if __name__ == '__main__':
    print("=" * 55)
    print("  ATTENTION SCORES & WEIGHTS")
    print("=" * 55)

    for sent in EXAMPLE_SENTENCES[:2]:
        r = compute_attention_matrix(sent)
        print(f"\n  '{sent}'")
        print(f"  {r['num_words']} words → {r['num_words']}x{r['num_words']} attention matrix")
        for t in r['top_attention']:
            top = t['attends_to'][0]
            print(f"    {t['word']:8s} → {top['word']:8s} ({top['weight']:.1%})")
