"""
=== MODULE 08.1: ONE-HOT ENCODING ===

Before a neural network can process words, we must convert them
to numbers. The simplest approach is ONE-HOT ENCODING:

  Vocabulary: [cat, dog, fish, bird, the, a, sat, ran]
  
  "cat"  = [1, 0, 0, 0, 0, 0, 0, 0]
  "dog"  = [0, 1, 0, 0, 0, 0, 0, 0]
  "fish" = [0, 0, 1, 0, 0, 0, 0, 0]
  "bird" = [0, 0, 0, 1, 0, 0, 0, 0]

  Each word gets a vector of length V (vocabulary size).
  All zeros except a single 1 at that word's index.

THIS WORKS BUT HAS PROBLEMS:

  1. NO MEANING: "cat" and "dog" are equally far apart as
     "cat" and "the". The encoding carries zero semantic info.
     distance(cat, dog) = sqrt(2)
     distance(cat, the) = sqrt(2)  — same!

  2. HUGE VECTORS: A vocabulary of 50,000 words means every
     word is a 50,000-dimensional vector with one 1 and
     49,999 zeros. Extremely wasteful.

  3. NO SIMILARITY: Similar words (run/ran, cat/kitten) have
     no relationship in one-hot space. The network must learn
     from scratch that these are related.

Module 08.2 introduces DENSE EMBEDDINGS — short vectors
(e.g., 128 numbers) where similar words are CLOSE TOGETHER
in the space. "cat" and "dog" would be nearby, while "cat"
and "the" would be far apart.
"""

import numpy as np


# ============================================================
# VOCABULARY & ONE-HOT ENCODING
# ============================================================
VOCAB = ['cat', 'dog', 'fish', 'bird', 'horse', 'the', 'a', 'an',
         'sat', 'ran', 'jumped', 'slept', 'on', 'under', 'near',
         'mat', 'bed', 'table', 'big', 'small', 'fast', 'slow',
         'happy', 'sad', 'red', 'blue', 'is', 'was', 'run', 'walk']

WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}


def one_hot_encode(word):
    """Convert a word to its one-hot vector."""
    idx = WORD_TO_IDX.get(word.lower(), -1)
    vec = [0] * len(VOCAB)
    if idx >= 0:
        vec[idx] = 1
    return {
        'word': word,
        'index': idx,
        'vector': vec,
        'vocab_size': len(VOCAB),
        'found': idx >= 0,
    }


def encode_sentence(sentence):
    """Encode each word in a sentence."""
    words = sentence.lower().split()
    return {
        'words': words,
        'encodings': [one_hot_encode(w) for w in words],
        'vocab_size': len(VOCAB),
    }


def compute_distances(words=None):
    """Compute Euclidean distances between all pairs of words.
    
    Shows that ALL one-hot distances are identical (sqrt(2)),
    regardless of semantic similarity.
    """
    if words is None:
        words = ['cat', 'dog', 'fish', 'the', 'sat', 'big', 'small', 'ran', 'run', 'happy']
    
    n = len(words)
    vectors = []
    for w in words:
        enc = one_hot_encode(w)
        vectors.append(enc['vector'])
    
    vectors = np.array(vectors, dtype=float)
    
    distances = []
    for i in range(n):
        row = []
        for j in range(n):
            d = float(np.sqrt(np.sum((vectors[i] - vectors[j])**2)))
            row.append(round(d, 3))
        distances.append(row)
    
    # Dot products (all zero for different words in one-hot)
    dot_products = []
    for i in range(n):
        row = []
        for j in range(n):
            dp = float(np.dot(vectors[i], vectors[j]))
            row.append(dp)
        distances.append(row)
    
    return {
        'words': words,
        'distances': distances[:n],  # just the distance matrix
        'dot_products': dot_products,
        'all_same': True,  # key insight
        'distance_value': round(float(np.sqrt(2)), 3),
    }


def compare_similar_words():
    """Show that semantically similar words have NO special relationship."""
    pairs = [
        ('cat', 'dog', 'Both are animals'),
        ('cat', 'the', 'Unrelated words'),
        ('big', 'small', 'Both are adjectives (opposites)'),
        ('big', 'fish', 'Unrelated'),
        ('ran', 'run', 'Same verb, different tense'),
        ('ran', 'table', 'Unrelated'),
        ('happy', 'sad', 'Both are emotions (opposites)'),
        ('happy', 'under', 'Unrelated'),
    ]
    
    results = []
    for w1, w2, relation in pairs:
        v1 = np.array(one_hot_encode(w1)['vector'], dtype=float)
        v2 = np.array(one_hot_encode(w2)['vector'], dtype=float)
        dist = float(np.sqrt(np.sum((v1 - v2)**2)))
        dot = float(np.dot(v1, v2))
        results.append({
            'word1': w1,
            'word2': w2,
            'relation': relation,
            'distance': round(dist, 4),
            'dot_product': dot,
            'same_distance': True,
        })
    
    return {
        'pairs': results,
        'insight': 'Every pair has distance sqrt(2) = 1.4142. One-hot encoding contains ZERO information about word similarity.',
    }


def show_sparsity(vocab_size=None):
    """Show how wasteful one-hot is for large vocabularies."""
    if vocab_size is None:
        vocab_size = len(VOCAB)
    
    sizes = [
        {'name': 'Our vocab', 'size': len(VOCAB), 'zeros_pct': round((1 - 1/len(VOCAB)) * 100, 2)},
        {'name': 'Small NLP', 'size': 10000, 'zeros_pct': round((1 - 1/10000) * 100, 4)},
        {'name': 'GPT-2', 'size': 50257, 'zeros_pct': round((1 - 1/50257) * 100, 4)},
        {'name': 'GPT-4 (est)', 'size': 100000, 'zeros_pct': round((1 - 1/100000) * 100, 4)},
    ]
    
    return {
        'sizes': sizes,
        'insight': 'At GPT-2 scale, each word vector is 99.998% zeros. Dense embeddings compress this to ~768 meaningful numbers.',
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  ONE-HOT ENCODING")
    print("=" * 55)
    
    for w in ['cat', 'dog', 'the']:
        enc = one_hot_encode(w)
        ones = sum(enc['vector'])
        print(f"  '{w}' -> index {enc['index']}, {ones} one, {len(enc['vector'])-1} zeros")
    
    print("\n  Distances (all pairs):")
    comp = compare_similar_words()
    for p in comp['pairs']:
        print(f"    {p['word1']:6s} <-> {p['word2']:6s}  dist={p['distance']}  ({p['relation']})")
