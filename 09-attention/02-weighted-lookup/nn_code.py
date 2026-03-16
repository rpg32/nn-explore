"""
=== MODULE 09.2: ATTENTION AS WEIGHTED LOOKUP ===

Module 09.1 showed the bottleneck: an RNN encoder compresses
a whole sentence into one vector. The decoder is blind to
everything except that one summary.

ATTENTION fixes this by letting the decoder LOOK BACK at every
encoder step and pick what's relevant.

THE REAL-WORLD ANALOGY:

  Imagine you're translating a French sentence to English.
  You're working on the 5th English word. Do you stare at
  a summary of the French sentence? No — you look back at
  the specific French words that are relevant to what you're
  writing NOW.

  That's attention: a selective, weighted lookup.

THE MECHANISM — Query, Key, Value:

  Think of it like a library search:

  QUERY (Q): "What am I looking for?"
    The decoder's current state — what it needs right now.
    "I'm trying to generate the next word in the translation."

  KEY (K): "What does each source word offer?"
    Each encoder hidden state advertises what it contains.
    "I'm the word 'chat' (cat). I'm the word 'sur' (on)."

  VALUE (V): "What information does each source word provide?"
    The actual content to retrieve if selected.
    Often the same as the key, or a transformed version.

  THE PROCESS:
    1. Compute a score for each key: score = Q · K
       "How well does my query match each key?"

    2. Softmax the scores into weights (sum to 1)
       "Turn scores into probabilities"

    3. Weighted sum of values: output = Σ(weight × value)
       "Blend the relevant information"

  The result: instead of one fixed summary vector, the decoder
  gets a CUSTOM summary for each output step, weighted by
  what's actually relevant.

THE KEY INSIGHT:
  Attention is just a WEIGHTED AVERAGE.
  The weights come from comparing what you need (query)
  with what's available (keys). High match = high weight.
  It's a soft lookup — not picking one item, but blending all
  items with relevance-based weights.
"""

import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ============================================================
# ATTENTION STEP BY STEP
# ============================================================
def attention_step_by_step(query_word, key_words, embed_dim=4, seed=42):
    """Walk through each step of the attention mechanism.

    Uses simple word embeddings to demonstrate Q, K, V matching.
    """
    np.random.seed(seed)

    all_words = list(set([query_word] + key_words))
    all_words.sort()
    w2i = {w: i for i, w in enumerate(all_words)}

    # Random embeddings (pretend these are learned)
    # We bias similar words to have similar embeddings
    CATEGORIES = {
        'cat': 'animal', 'dog': 'animal', 'bird': 'animal', 'fish': 'animal',
        'sat': 'verb', 'ran': 'verb', 'jumped': 'verb', 'flew': 'verb',
        'the': 'func', 'a': 'func', 'on': 'func', 'over': 'func',
        'big': 'adj', 'small': 'adj', 'fast': 'adj', 'happy': 'adj',
        'mat': 'noun', 'rug': 'noun', 'tree': 'noun', 'house': 'noun',
    }
    CAT_VECS = {
        'animal': np.array([1.0, 0.5, -0.3, 0.8]),
        'verb':   np.array([-0.5, 1.0, 0.7, -0.2]),
        'func':   np.array([0.1, -0.1, 0.1, -0.1]),
        'adj':    np.array([0.3, -0.8, 1.0, 0.4]),
        'noun':   np.array([0.7, 0.3, -0.5, 0.9]),
    }

    embeddings = {}
    for w in all_words:
        cat = CATEGORIES.get(w, 'func')
        base = CAT_VECS.get(cat, np.zeros(embed_dim))
        noise = np.random.randn(embed_dim) * 0.15
        embeddings[w] = base + noise

    # W_Q, W_K, W_V projection matrices (learned in real attention)
    W_Q = np.eye(embed_dim) + np.random.randn(embed_dim, embed_dim) * 0.1
    W_K = np.eye(embed_dim) + np.random.randn(embed_dim, embed_dim) * 0.1
    W_V = np.eye(embed_dim) + np.random.randn(embed_dim, embed_dim) * 0.1

    # Step 1: Compute Q, K, V
    q = W_Q @ embeddings[query_word]
    keys = [W_K @ embeddings[w] for w in key_words]
    values = [W_V @ embeddings[w] for w in key_words]

    # Step 2: Compute scores (dot product of query with each key)
    d_k = embed_dim
    scores = [float(np.dot(q, k) / np.sqrt(d_k)) for k in keys]

    # Step 3: Softmax to get weights
    scores_arr = np.array(scores)
    weights = softmax(scores_arr).tolist()

    # Step 4: Weighted sum of values
    output = np.zeros(embed_dim)
    for i in range(len(values)):
        output += weights[i] * values[i]

    return {
        'query_word': query_word,
        'key_words': key_words,

        'step1_embed': {
            'query_embed': [round(float(v), 3) for v in embeddings[query_word]],
            'key_embeds': {w: [round(float(v), 3) for v in embeddings[w]] for w in key_words},
        },
        'step2_qkv': {
            'query': [round(float(v), 3) for v in q],
            'keys': {w: [round(float(v), 3) for v in keys[i]] for i, w in enumerate(key_words)},
            'values': {w: [round(float(v), 3) for v in values[i]] for i, w in enumerate(key_words)},
        },
        'step3_scores': {w: round(scores[i], 3) for i, w in enumerate(key_words)},
        'step4_weights': {w: round(weights[i], 3) for i, w in enumerate(key_words)},
        'step5_output': [round(float(v), 3) for v in output],

        'categories': {w: CATEGORIES.get(w, '?') for w in [query_word] + key_words},
    }


# ============================================================
# PREDEFINED EXAMPLES
# ============================================================
EXAMPLES = {
    'animal_lookup': {
        'name': 'Looking for an animal',
        'query': 'cat',
        'keys': ['the', 'big', 'dog', 'sat', 'on', 'the', 'mat'],
        'desc': '"cat" as query should attend most to "dog" (similar category)',
    },
    'verb_lookup': {
        'name': 'Looking for a verb',
        'query': 'ran',
        'keys': ['the', 'cat', 'sat', 'on', 'the', 'big', 'rug'],
        'desc': '"ran" should attend most to "sat" (both are verbs)',
    },
    'adj_lookup': {
        'name': 'Looking for an adjective',
        'query': 'big',
        'keys': ['the', 'small', 'cat', 'ran', 'fast', 'on', 'mat'],
        'desc': '"big" should attend most to "small" and "fast" (adjectives)',
    },
    'noun_lookup': {
        'name': 'Looking for a noun',
        'query': 'mat',
        'keys': ['the', 'cat', 'sat', 'on', 'the', 'rug', 'tree'],
        'desc': '"mat" should attend most to "rug" and "tree" (nouns/objects)',
    },
}


def run_example(example_name, seed=42):
    ex = EXAMPLES.get(example_name, EXAMPLES['animal_lookup'])
    # Deduplicate keys for cleaner display
    seen = set()
    unique_keys = []
    for w in ex['keys']:
        if w not in seen:
            unique_keys.append(w)
            seen.add(w)
    result = attention_step_by_step(ex['query'], unique_keys, seed=seed)
    result['example_name'] = ex['name']
    result['example_desc'] = ex['desc']
    return result


if __name__ == '__main__':
    print("=" * 55)
    print("  ATTENTION AS WEIGHTED LOOKUP")
    print("=" * 55)

    for name in EXAMPLES:
        r = run_example(name)
        print(f"\n  {r['example_name']}: query='{r['query_word']}'")
        sorted_w = sorted(r['step4_weights'].items(), key=lambda x: -x[1])
        for w, wt in sorted_w:
            bar = '#' * int(wt * 40)
            print(f"    {w:6s}: {wt:.3f} {bar}")
