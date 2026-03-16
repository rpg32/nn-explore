"""
=== MODULE 07.1: WHY SEQUENCES ARE SPECIAL ===

Before diving into RNNs, we need to understand WHY we need
special architectures for sequential data. The core problem:

  AN MLP TREATS ALL INPUTS SIMULTANEOUSLY.
  IT HAS NO CONCEPT OF ORDER.

Consider these two sentences:
  "dog bites man"   → mundane news
  "man bites dog"   → front-page headline!

Same three words, completely different meaning. But if we
feed them to an MLP by summing or averaging word vectors,
both sentences produce THE EXACT SAME representation.

  bag("dog bites man") = vec("dog") + vec("bites") + vec("man")
  bag("man bites dog") = vec("man") + vec("bites") + vec("dog")

Addition is commutative, so these are identical!

The same problem appears everywhere:
  "not good"  vs  "good"       — negation changes meaning
  "hot cold"  vs  "cold hot"   — temperature trajectory matters
  [1,2,3,4,5] vs [5,4,3,2,1]  — rising vs falling signal

For language, music, stock prices, and any time-varying data,
ORDER IS EVERYTHING. We need a network architecture that
processes elements one at a time, building up understanding
as it goes. That's what RNNs do — next module!
"""

import numpy as np


# ============================================================
# FIXED WORD EMBEDDINGS (random but deterministic)
# ============================================================
# We assign each word a fixed random vector. In real NLP,
# these would be learned (Word2Vec, GloVe), but random
# embeddings are enough to demonstrate the order problem.

EMBED_DIM = 8  # Each word becomes an 8-dimensional vector

def get_embedding(word, seed_offset=0):
    """Get a fixed random embedding for a word.

    Uses the word's hash as a seed so the same word
    always gets the same vector, but different words
    get different vectors.
    """
    # Deterministic: same word → same vector every time
    seed = abs(hash(word)) % (2**31)
    rng = np.random.RandomState(seed + seed_offset)
    return rng.randn(EMBED_DIM) * 0.5


# Pre-compute embeddings for our demo vocabulary
VOCAB = ['dog', 'bites', 'man', 'cat', 'chases', 'mouse',
         'not', 'good', 'very', 'bad', 'the', 'a',
         'hot', 'cold', 'warm', 'is', 'was', 'big', 'small']

EMBEDDINGS = {word: get_embedding(word) for word in VOCAB}


# ============================================================
# SIMPLE MLP (bag-of-words — ignores order)
# ============================================================
class BagOfWordsMLP:
    """A simple MLP that processes sentences as bags of words.

    Step 1: Look up embedding for each word
    Step 2: SUM all embeddings (this is where order is lost!)
    Step 3: Feed the sum through a hidden layer + output layer

    Because addition is commutative (a+b = b+a), rearranging
    the words produces the EXACT SAME sum → EXACT SAME output.
    """

    def __init__(self, embed_dim=EMBED_DIM, hidden_size=16, output_size=4, seed=42):
        np.random.seed(seed)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        # Layer 1: sum of embeddings → hidden
        self.W1 = np.random.randn(hidden_size, embed_dim) * np.sqrt(2.0 / embed_dim)
        self.b1 = np.zeros(hidden_size)

        # Layer 2: hidden → output
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, words):
        """Process a list of words through the bag-of-words MLP.

        Returns all intermediate values so we can visualize the problem.
        """
        # Step 1: Look up embeddings
        embeddings = []
        for w in words:
            w_lower = w.lower()
            if w_lower in EMBEDDINGS:
                embeddings.append(EMBEDDINGS[w_lower].copy())
            else:
                embeddings.append(get_embedding(w_lower))

        # Step 2: SUM all embeddings → bag-of-words representation
        # THIS IS WHERE ORDER INFORMATION IS DESTROYED
        bag = np.sum(embeddings, axis=0)

        # Step 3: Hidden layer with ReLU
        z1 = self.W1 @ bag + self.b1
        h = np.maximum(0, z1)  # ReLU

        # Step 4: Output layer with softmax
        z2 = self.W2 @ h + self.b2
        exp_z = np.exp(z2 - z2.max())
        probs = exp_z / exp_z.sum()

        return {
            'words': words,
            'embeddings': [e.tolist() for e in embeddings],
            'bag_vector': bag.tolist(),
            'hidden_pre': z1.tolist(),
            'hidden_post': h.tolist(),
            'output_logits': z2.tolist(),
            'output_probs': probs.tolist(),
        }


# ============================================================
# CORE DEMOS
# ============================================================

# Shared MLP instance (same weights for all comparisons)
mlp = BagOfWordsMLP()


def classify_with_mlp(sentence):
    """Feed a sentence through the bag-of-words MLP.

    Returns the full computation trace showing that
    order doesn't affect the output.
    """
    words = sentence.strip().split()
    return mlp.forward(words)


def order_experiment():
    """Run the key experiment: same words, different order → same output.

    This is the central demonstration of WHY we need RNNs.
    We show multiple pairs of sentences where rearranging words
    changes the meaning but the MLP can't tell the difference.
    """
    pairs = [
        {
            'name': 'Subject vs Object',
            'sentence_a': 'dog bites man',
            'sentence_b': 'man bites dog',
            'meaning_a': 'A dog bit a person — routine event',
            'meaning_b': 'A person bit a dog — bizarre headline!',
        },
        {
            'name': 'Negation',
            'sentence_a': 'not good',
            'sentence_b': 'good not',
            'meaning_a': 'Negative sentiment',
            'meaning_b': 'Nonsense, but MLP sees it the same',
        },
        {
            'name': 'Degree Modifier',
            'sentence_a': 'very good not bad',
            'sentence_b': 'not very good bad',
            'meaning_a': 'Positive sentiment',
            'meaning_b': 'Negative sentiment',
        },
        {
            'name': 'Temperature Trajectory',
            'sentence_a': 'cold warm hot',
            'sentence_b': 'hot warm cold',
            'meaning_a': 'Getting warmer — heating up',
            'meaning_b': 'Getting cooler — cooling down',
        },
    ]

    results = []
    for pair in pairs:
        result_a = mlp.forward(pair['sentence_a'].split())
        result_b = mlp.forward(pair['sentence_b'].split())

        # Check if outputs are identical
        bag_match = np.allclose(result_a['bag_vector'], result_b['bag_vector'])
        output_match = np.allclose(result_a['output_probs'], result_b['output_probs'])

        results.append({
            'name': pair['name'],
            'sentence_a': pair['sentence_a'],
            'sentence_b': pair['sentence_b'],
            'meaning_a': pair['meaning_a'],
            'meaning_b': pair['meaning_b'],
            'result_a': result_a,
            'result_b': result_b,
            'bag_vectors_identical': bag_match,
            'outputs_identical': output_match,
        })

    return results


def time_series_demo():
    """Demonstrate why order matters in numeric sequences.

    Show that different time series patterns (rising, falling,
    oscillating) have the same average/sum but completely
    different meanings.
    """
    patterns = {
        'rising': {
            'name': 'Rising',
            'description': 'Steadily increasing — growth, warming, acceleration',
            'values': [1, 2, 3, 4, 5, 6, 7, 8],
        },
        'falling': {
            'name': 'Falling',
            'description': 'Steadily decreasing — decline, cooling, deceleration',
            'values': [8, 7, 6, 5, 4, 3, 2, 1],
        },
        'peak': {
            'name': 'Peak',
            'description': 'Goes up then down — bubble, noon temperature',
            'values': [1, 3, 5, 8, 8, 5, 3, 1],
        },
        'valley': {
            'name': 'Valley',
            'description': 'Goes down then up — recession, winter low',
            'values': [8, 5, 3, 1, 1, 3, 5, 8],
        },
        'oscillating': {
            'name': 'Oscillating',
            'description': 'Alternating high and low — vibration, AC current',
            'values': [1, 8, 1, 8, 1, 8, 1, 8],
        },
    }

    results = {}
    for key, pat in patterns.items():
        vals = np.array(pat['values'], dtype=float)
        results[key] = {
            'name': pat['name'],
            'description': pat['description'],
            'values': pat['values'],
            'sum': float(vals.sum()),
            'mean': float(vals.mean()),
            'sorted': sorted(pat['values']),
        }

    # Highlight: rising and falling have the same sum, mean, and sorted values!
    # An order-blind model literally cannot tell them apart.
    return {
        'patterns': results,
        'key_insight': (
            'Rising and Falling have identical sums (36), identical means (4.5), '
            'and identical sorted values. An MLP that sums or averages its inputs '
            'CANNOT distinguish them. The entire meaning is in the ORDER.'
        ),
    }


def compare_sentences(sentence_a, sentence_b):
    """Compare two user-provided sentences through the MLP.

    Shows whether the MLP produces the same or different outputs.
    """
    result_a = classify_with_mlp(sentence_a)
    result_b = classify_with_mlp(sentence_b)

    bag_diff = float(np.linalg.norm(
        np.array(result_a['bag_vector']) - np.array(result_b['bag_vector'])
    ))
    output_diff = float(np.linalg.norm(
        np.array(result_a['output_probs']) - np.array(result_b['output_probs'])
    ))

    # Check if they share the same set of words (regardless of order)
    set_a = set(w.lower() for w in sentence_a.split())
    set_b = set(w.lower() for w in sentence_b.split())
    same_words = set_a == set_b

    return {
        'sentence_a': sentence_a,
        'sentence_b': sentence_b,
        'result_a': result_a,
        'result_b': result_b,
        'same_word_set': same_words,
        'bag_difference': bag_diff,
        'output_difference': output_diff,
        'bag_identical': bag_diff < 1e-10,
        'output_identical': output_diff < 1e-10,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  WHY SEQUENCES ARE SPECIAL")
    print("=" * 60)
    print()

    # Demo 1: Order experiment
    print("  ORDER EXPERIMENT:")
    print("  -" * 30)
    results = order_experiment()
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    A: '{r['sentence_a']}' — {r['meaning_a']}")
        print(f"    B: '{r['sentence_b']}' — {r['meaning_b']}")
        print(f"    Bag vectors identical? {r['bag_vectors_identical']}")
        print(f"    MLP outputs identical? {r['outputs_identical']}")

    # Demo 2: Time series
    print("\n\n  TIME SERIES PATTERNS:")
    print("  -" * 30)
    ts = time_series_demo()
    for key, pat in ts['patterns'].items():
        print(f"    {pat['name']:12s}: {pat['values']}  sum={pat['sum']:.0f}  mean={pat['mean']:.1f}")
    print(f"\n  Insight: {ts['key_insight']}")
