"""
=== MODULE 08.2: DENSE EMBEDDINGS ===

One-hot encoding (Module 08.1) has two problems:
  1. Huge sparse vectors (50,000 dims with one 1)
  2. No similarity info (cat and dog are as far apart as cat and the)

DENSE EMBEDDINGS fix both:

  Instead of a sparse vector with one 1, each word gets a SHORT
  vector of LEARNED numbers (e.g., 2-300 dimensions):

    "cat"  = [0.82, -0.41, 0.13, ...]   (128 numbers)
    "dog"  = [0.79, -0.38, 0.15, ...]   (128 numbers — CLOSE to cat!)
    "the"  = [-0.21, 0.67, -0.55, ...]  (128 numbers — FAR from cat)

  Similar words have similar vectors.
  The DISTANCE between vectors encodes MEANING.

HOW TO THINK ABOUT IT:

  Imagine a 2D space where you place words:
    - Animals cluster together (cat, dog, fish, bird)
    - Verbs cluster together (ran, jumped, sat)
    - Adjectives cluster together (big, small, fast, slow)

  In reality, embeddings use 128-768 dimensions, not 2.
  But the principle is the same: meaning = geometry.

WHAT THE DIMENSIONS MEAN:

  No single dimension has a clear meaning like "animal-ness".
  Instead, meaning emerges from the PATTERN across all dimensions.
  It's like how RGB values don't individually mean much, but
  together they specify a precise color.

  However, directions in the space CAN encode relationships:
    king - man + woman ≈ queen
  This famous example shows that gender is a direction,
  royalty is a direction, and you can do algebra with them.

WHERE DO EMBEDDINGS COME FROM?

  They're LEARNED by training (Module 08.3). The embedding for
  each word starts random and is adjusted by backpropagation
  until words that appear in similar contexts end up nearby.
"""

import numpy as np


# ============================================================
# PRE-BUILT EMBEDDINGS (simulated, 2D for visualization)
# ============================================================
# In reality these would be learned. We hand-craft 2D positions
# so similar words cluster together for clear visualization.

EMBEDDINGS_2D = {
    # Animals (cluster top-right)
    'cat':    [2.1, 2.3],
    'dog':    [2.4, 2.0],
    'fish':   [3.0, 2.8],
    'bird':   [2.8, 3.2],
    'horse':  [1.8, 1.6],
    'kitten': [2.2, 2.5],
    'puppy':  [2.5, 2.2],

    # Verbs (cluster bottom-left)
    'ran':     [-1.5, -1.8],
    'run':     [-1.3, -1.6],
    'walked':  [-1.8, -1.2],
    'jumped':  [-1.0, -2.2],
    'sat':     [-2.0, -0.8],
    'slept':   [-2.3, -0.5],

    # Adjectives (cluster left-middle)
    'big':    [-2.5, 1.5],
    'small':  [-2.3, 1.2],
    'large':  [-2.4, 1.7],
    'tiny':   [-2.1, 1.0],
    'fast':   [-1.5, 2.0],
    'slow':   [-1.8, 1.8],
    'happy':  [-0.5, 2.5],
    'sad':    [-0.3, 2.2],

    # Function words (cluster center-bottom)
    'the':   [0.2, -0.5],
    'a':     [0.4, -0.3],
    'an':    [0.3, -0.4],
    'on':    [0.8, -0.8],
    'under': [0.6, -1.0],
    'near':  [0.7, -0.6],
    'is':    [0.0, -1.2],
    'was':   [0.1, -1.4],

    # Nouns - objects (cluster right-bottom)
    'mat':   [1.5, -0.5],
    'bed':   [1.8, -0.2],
    'table': [1.3, -0.8],
    'house': [1.6, -1.0],
    'car':   [1.0, -1.5],

    # Royalty (for the king-queen demo)
    'king':   [3.5, -1.0],
    'queen':  [3.5, 0.5],
    'man':    [1.5, -1.5],
    'woman':  [1.5, 0.0],
    'prince': [3.2, -0.8],
    'princess': [3.2, 0.7],
}

# Categories for coloring
CATEGORIES = {
    'animal': ['cat', 'dog', 'fish', 'bird', 'horse', 'kitten', 'puppy'],
    'verb': ['ran', 'run', 'walked', 'jumped', 'sat', 'slept'],
    'adjective': ['big', 'small', 'large', 'tiny', 'fast', 'slow', 'happy', 'sad'],
    'function': ['the', 'a', 'an', 'on', 'under', 'near', 'is', 'was'],
    'object': ['mat', 'bed', 'table', 'house', 'car'],
    'royalty': ['king', 'queen', 'man', 'woman', 'prince', 'princess'],
}

CATEGORY_COLORS = {
    'animal': '#4fc3f7',
    'verb': '#ef5350',
    'adjective': '#66bb6a',
    'function': '#888888',
    'object': '#ffb74d',
    'royalty': '#ab47bc',
}


def get_word_category(word):
    for cat, words in CATEGORIES.items():
        if word in words:
            return cat
    return 'unknown'


def get_all_embeddings():
    """Return all words with their 2D positions and categories."""
    words = []
    for word, pos in EMBEDDINGS_2D.items():
        words.append({
            'word': word,
            'x': pos[0],
            'y': pos[1],
            'category': get_word_category(word),
            'color': CATEGORY_COLORS.get(get_word_category(word), '#555'),
        })
    return {
        'words': words,
        'categories': {cat: CATEGORY_COLORS[cat] for cat in CATEGORIES},
    }


def compute_nearest(word, n=5):
    """Find the N nearest words to a given word."""
    if word not in EMBEDDINGS_2D:
        return {'error': f'"{word}" not in vocabulary'}

    target = np.array(EMBEDDINGS_2D[word])
    distances = []
    for w, pos in EMBEDDINGS_2D.items():
        if w == word:
            continue
        d = float(np.sqrt(np.sum((target - np.array(pos))**2)))
        distances.append({'word': w, 'distance': round(d, 4), 'category': get_word_category(w)})

    distances.sort(key=lambda x: x['distance'])
    return {
        'query': word,
        'query_category': get_word_category(word),
        'nearest': distances[:n],
        'farthest': distances[-3:],
    }


def analogy(a, b, c):
    """Compute a - b + c ≈ ? (e.g., king - man + woman ≈ queen)"""
    if a not in EMBEDDINGS_2D or b not in EMBEDDINGS_2D or c not in EMBEDDINGS_2D:
        return {'error': 'Word not in vocabulary'}

    va = np.array(EMBEDDINGS_2D[a])
    vb = np.array(EMBEDDINGS_2D[b])
    vc = np.array(EMBEDDINGS_2D[c])

    result_vec = va - vb + vc

    # Find nearest word to result
    distances = []
    for w, pos in EMBEDDINGS_2D.items():
        if w in [a, b, c]:
            continue
        d = float(np.sqrt(np.sum((result_vec - np.array(pos))**2)))
        distances.append({'word': w, 'distance': round(d, 4)})

    distances.sort(key=lambda x: x['distance'])

    return {
        'a': a, 'b': b, 'c': c,
        'formula': f'{a} - {b} + {c}',
        'result_position': [float(result_vec[0]), float(result_vec[1])],
        'nearest_word': distances[0]['word'],
        'top5': distances[:5],
        'positions': {
            a: EMBEDDINGS_2D[a],
            b: EMBEDDINGS_2D[b],
            c: EMBEDDINGS_2D[c],
        },
    }


def compare_distances(pairs):
    """Compare distances between word pairs — show that embeddings encode similarity."""
    results = []
    for w1, w2 in pairs:
        if w1 in EMBEDDINGS_2D and w2 in EMBEDDINGS_2D:
            v1 = np.array(EMBEDDINGS_2D[w1])
            v2 = np.array(EMBEDDINGS_2D[w2])
            d = float(np.sqrt(np.sum((v1 - v2)**2)))
            results.append({
                'word1': w1, 'word2': w2,
                'distance': round(d, 4),
                'cat1': get_word_category(w1),
                'cat2': get_word_category(w2),
                'same_category': get_word_category(w1) == get_word_category(w2),
            })
    return {'pairs': results}


DEFAULT_PAIRS = [
    ['cat', 'dog'],
    ['cat', 'the'],
    ['big', 'small'],
    ['big', 'fish'],
    ['ran', 'run'],
    ['ran', 'table'],
    ['king', 'queen'],
    ['king', 'car'],
]


if __name__ == '__main__':
    print("=" * 55)
    print("  DENSE EMBEDDINGS")
    print("=" * 55)

    print("\n  Nearest neighbors:")
    for w in ['cat', 'ran', 'big']:
        nn = compute_nearest(w, 3)
        nns = ', '.join(f"{n['word']}({n['distance']:.2f})" for n in nn['nearest'])
        print(f"    {w}: {nns}")

    print("\n  Distances (embedding vs one-hot):")
    for w1, w2 in DEFAULT_PAIRS:
        v1 = np.array(EMBEDDINGS_2D[w1])
        v2 = np.array(EMBEDDINGS_2D[w2])
        d = np.sqrt(np.sum((v1 - v2)**2))
        print(f"    {w1:6s} <-> {w2:6s}  embed_dist={d:.3f}  (one-hot was always 1.414)")

    print("\n  Analogy: king - man + woman =", analogy('king', 'man', 'woman')['nearest_word'])
