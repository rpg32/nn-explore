"""
=== MODULE 08.3: LEARNING EMBEDDINGS ===

Module 08.2 showed that dense embeddings place similar words close
together. But WHERE do those numbers come from? They're LEARNED
from data, using a beautifully simple idea:

THE KEY INSIGHT:
  "You shall know a word by the company it keeps." (J.R. Firth, 1957)

  Words that appear in similar contexts have similar meanings:
    "The ___ sat on the mat."     → cat, dog, bird (animals)
    "She ___ quickly to the store." → ran, walked, drove (motion verbs)

  Words that fill the same blanks are semantically similar.

HOW WORD2VEC WORKS (Skip-gram, simplified):

  1. Start with RANDOM embeddings for every word
  2. Slide a window over the training text
  3. For each center word, predict the surrounding context words
  4. Use backprop to update the embeddings
  5. Words that predict the same contexts get pushed CLOSER together

  Example: In "the cat sat on the mat"
    center="cat", context=["the", "sat"]
    center="sat", context=["cat", "on"]

  The network learns: "cat" and "dog" both predict similar context
  words ("the", "sat", "on"), so their embeddings converge.

THE TRAINING LOOP:
  Input:  one-hot vector for center word (e.g., "cat")
  Hidden: multiply by embedding matrix → get the embedding vector
  Output: predict context words using another matrix
  Loss:   how well did we predict the actual context?
  Update: adjust embedding matrix via backprop

  The hidden layer IS the embedding. After training,
  row i of the embedding matrix = the embedding for word i.

  This is just an MLP with one hidden layer! The trick is the
  TRAINING TASK (predict context), not the architecture.
"""

import numpy as np


# ============================================================
# TINY CORPUS FOR TRAINING
# ============================================================
CORPUS = """
the cat sat on the mat . the dog sat on the rug .
the cat chased the bird . the dog chased the cat .
a big cat sat on a big mat . a small dog sat on a small rug .
the happy cat ran fast . the sad dog walked slow .
a fast bird flew over the big house . a slow fish swam under the small bridge .
the cat and the dog are friends . the bird and the fish are not friends .
the big dog ran to the small cat . the fast cat ran from the slow dog .
a happy bird sang on the mat . a sad fish sat under the bridge .
the cat sat . the dog sat . the bird flew . the fish swam .
""".strip()


def build_vocab(text):
    words = text.lower().split()
    vocab = sorted(set(words))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return words, vocab, w2i, i2w


WORDS, VOCAB, W2I, I2W = build_vocab(CORPUS)
VOCAB_SIZE = len(VOCAB)


# ============================================================
# SKIP-GRAM TRAINING
# ============================================================
def generate_training_pairs(words, w2i, window=2):
    """Generate (center_word, context_word) pairs from text."""
    pairs = []
    for i, word in enumerate(words):
        center = w2i[word]
        for j in range(max(0, i - window), min(len(words), i + window + 1)):
            if j != i:
                context = w2i[words[j]]
                pairs.append((center, context))
    return pairs


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class Word2VecSkipGram:
    """Minimal Word2Vec skip-gram model.

    Architecture: center_word → embedding → predict_context_word
    This is just an MLP: input(V) → hidden(D) → output(V)
    The hidden layer weights ARE the embeddings.
    """

    def __init__(self, vocab_size, embed_dim=2, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # W_embed: the EMBEDDING MATRIX (vocab_size x embed_dim)
        # Row i = embedding for word i
        # This IS the result — after training, these are the word vectors
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.3

        # W_context: the output matrix (embed_dim x vocab_size)
        # Used to predict context words from embeddings
        self.W_context = np.random.randn(embed_dim, vocab_size) * 0.3

        self.train_history = []

    def forward(self, center_idx):
        """Forward pass: center word → embedding → context prediction."""
        # Look up embedding (this replaces the one-hot × matrix multiply)
        embed = self.W_embed[center_idx]  # (embed_dim,)

        # Predict context word
        logits = self.W_context.T @ embed  # (vocab_size,)
        probs = softmax(logits)

        return embed, logits, probs

    def train_step(self, center_idx, context_idx, lr=0.05):
        """One training step: predict context from center, update weights."""
        embed, logits, probs = self.forward(center_idx)

        # Loss: cross-entropy
        loss = -np.log(probs[context_idx] + 1e-10)

        # Gradient at output: probs - one_hot(context)
        dlogits = probs.copy()
        dlogits[context_idx] -= 1  # (vocab_size,)

        # Gradient for W_context
        dW_context = np.outer(embed, dlogits)  # (embed_dim, vocab_size)

        # Gradient for embedding
        dembed = self.W_context @ dlogits  # (embed_dim,)

        # Update
        self.W_context -= lr * dW_context
        self.W_embed[center_idx] -= lr * dembed

        return loss

    def train_epoch(self, pairs, lr=0.05):
        """Train one pass through all pairs."""
        np.random.shuffle(pairs)
        total_loss = 0
        for center, context in pairs:
            total_loss += self.train_step(center, context, lr)
        avg_loss = total_loss / len(pairs)
        self.train_history.append(avg_loss)
        return avg_loss

    def get_embeddings(self, i2w):
        """Return all word embeddings for visualization."""
        results = []
        for i in range(self.vocab_size):
            results.append({
                'word': i2w[i],
                'x': float(self.W_embed[i, 0]),
                'y': float(self.W_embed[i, 1]) if self.embed_dim >= 2 else 0,
            })
        return results

    def get_nearest(self, word_idx, n=5):
        """Find nearest neighbors by cosine similarity."""
        target = self.W_embed[word_idx]
        norms = np.sqrt(np.sum(self.W_embed**2, axis=1))
        norms[norms == 0] = 1e-10
        similarities = (self.W_embed @ target) / (norms * np.linalg.norm(target) + 1e-10)
        top = np.argsort(similarities)[::-1]
        return [(int(i), float(similarities[i])) for i in top[:n+1] if i != word_idx][:n]


def train_embeddings(epochs=100, embed_dim=2, lr=0.1, seed=42):
    """Train word2vec and return embeddings at several checkpoints."""
    pairs = generate_training_pairs(WORDS, W2I, window=2)
    model = Word2VecSkipGram(VOCAB_SIZE, embed_dim, seed)

    checkpoints = []
    checkpoint_epochs = [0, 1, 5, 10, 25, 50, 100, 200, 300, 500]

    # Save initial (random) state
    checkpoints.append({
        'epoch': 0,
        'loss': None,
        'embeddings': model.get_embeddings(I2W),
    })

    for e in range(1, epochs + 1):
        loss = model.train_epoch(pairs, lr=lr)

        if e in checkpoint_epochs:
            checkpoints.append({
                'epoch': e,
                'loss': round(loss, 4),
                'embeddings': model.get_embeddings(I2W),
            })

    # Final state if not already captured
    if epochs not in checkpoint_epochs:
        checkpoints.append({
            'epoch': epochs,
            'loss': round(model.train_history[-1], 4),
            'embeddings': model.get_embeddings(I2W),
        })

    # Nearest neighbors at final state
    nearest = {}
    for w in ['cat', 'dog', 'big', 'small', 'sat', 'ran', 'the', 'bird']:
        if w in W2I:
            nn = model.get_nearest(W2I[w], 5)
            nearest[w] = [{'word': I2W[i], 'similarity': round(s, 3)} for i, s in nn]

    return {
        'checkpoints': checkpoints,
        'loss_history': [round(l, 4) for l in model.train_history],
        'nearest': nearest,
        'vocab': VOCAB,
        'vocab_size': VOCAB_SIZE,
        'num_pairs': len(pairs),
        'corpus_preview': CORPUS[:300],
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  LEARNING EMBEDDINGS (Word2Vec)")
    print("=" * 55)

    r = train_embeddings(epochs=200, embed_dim=2)
    print(f"  Vocab: {r['vocab_size']} words, {r['num_pairs']} training pairs")
    print(f"  Final loss: {r['loss_history'][-1]}")
    print(f"  Checkpoints: {[c['epoch'] for c in r['checkpoints']]}")
    print("\n  Nearest neighbors:")
    for w, nn in r['nearest'].items():
        nns = ', '.join(f"{n['word']}({n['similarity']:.2f})" for n in nn[:3])
        print(f"    {w}: {nns}")
