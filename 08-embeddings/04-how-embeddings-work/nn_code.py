"""
=== MODULE 08.4: HOW EMBEDDINGS ACTUALLY WORK ===

This module ties together everything from 08.1-08.3 and connects
to how real-world models use embeddings.

THE ARCHITECTURE (Word2Vec Skip-gram):

  It's a simple MLP:  [vocab_size, embed_dim, vocab_size]

  Example with vocab=8, embed_dim=2:

    Input layer:   8 neurons (one-hot encoded word)
    Hidden layer:  2 neurons (the embedding!)
    Output layer:  8 neurons (predicted context word probabilities)

  The hidden layer weight matrix IS the embedding table:

    W_embed = 8 rows × 2 columns

    Row 0 = embedding for word 0 ("cat")  = [0.82, -0.41]
    Row 1 = embedding for word 1 ("dog")  = [0.79, -0.38]
    Row 2 = embedding for word 2 ("the")  = [-0.21, 0.67]
    ...

  Multiplying a one-hot vector by this matrix just SELECTS a row.
  That's all an embedding lookup is — selecting a row from a table.

HOW TRAINING UPDATES THE EMBEDDINGS:

  Training pair: center="cat", context="sat"
  1. Feed one-hot("cat") through network → predicts context
  2. Loss = how wrong was the prediction of "sat"?
  3. Backprop adjusts W_embed row for "cat" to predict "sat" better
  4. Next pair: center="dog", context="sat"
  5. Backprop adjusts W_embed row for "dog" to predict "sat" better
  6. Both "cat" and "dog" rows move in the same direction → converge!

HOW REAL MODELS USE EMBEDDINGS:

  Word2Vec (2013): Standalone training, then used as features.
    - Train on billions of words of text
    - Get 300-dimensional vectors for ~3 million words
    - Use these vectors as input to other models

  GPT/BERT/Modern transformers: Embeddings are PART of the model.
    - The embedding matrix is the FIRST layer of the network
    - It's trained end-to-end with the rest of the model
    - Not a separate step — the embeddings are optimized for
      the specific task (language modeling, translation, etc.)

  In GPT-2:
    - Vocab size: 50,257 tokens
    - Embedding dim: 768
    - Embedding matrix: 50,257 × 768 = 38.6 million parameters
    - That's just the first layer! The rest of the model is bigger.

  The embedding matrix is literally a giant lookup table:
    token_id → row of the matrix → 768 numbers

  These 768 numbers then flow into the transformer layers
  (attention + feed-forward) where they get refined.
"""

import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ============================================================
# WORD2VEC NETWORK — STEP BY STEP
# ============================================================
VOCAB = ['cat', 'dog', 'sat', 'the', 'on', 'mat', 'ran', 'big']
W2I = {w: i for i, w in enumerate(VOCAB)}
I2W = {i: w for i, w in enumerate(VOCAB)}
V = len(VOCAB)


class Word2VecNetwork:
    """The actual neural network that produces embeddings."""

    def __init__(self, vocab_size, embed_dim=2, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # THE EMBEDDING MATRIX — this is what we're training to get
        # Shape: (vocab_size, embed_dim)
        # Row i = the embedding vector for word i
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.5

        # The output/prediction matrix
        # Shape: (embed_dim, vocab_size)
        self.W_output = np.random.randn(embed_dim, vocab_size) * 0.5

        self.history = []

    def forward_detailed(self, center_idx):
        """Full forward pass with all intermediates shown."""
        # Step 1: One-hot encode the center word
        one_hot = np.zeros(self.vocab_size)
        one_hot[center_idx] = 1.0

        # Step 2: Multiply by embedding matrix = select a row
        # This is the KEY insight: one_hot × W_embed = row[center_idx]
        embedding = self.W_embed[center_idx].copy()
        # Equivalently: embedding = one_hot @ self.W_embed

        # Step 3: Multiply by output matrix = get raw scores
        logits = self.W_output.T @ embedding  # (vocab_size,)

        # Step 4: Softmax = probabilities
        probs = softmax(logits)

        return {
            'one_hot': one_hot.tolist(),
            'embedding': embedding.tolist(),
            'logits': [round(float(v), 4) for v in logits],
            'probs': [round(float(v), 4) for v in probs],
            'W_embed_row': embedding.tolist(),
        }

    def train_step_detailed(self, center_idx, context_idx, lr=0.1):
        """One training step with full gradient trace."""
        # Forward
        embedding = self.W_embed[center_idx].copy()
        logits = self.W_output.T @ embedding
        probs = softmax(logits)

        # Loss
        loss = -np.log(probs[context_idx] + 1e-10)

        # Backward
        # Gradient at output layer
        d_logits = probs.copy()
        d_logits[context_idx] -= 1.0

        # Gradient for W_output
        d_W_output = np.outer(embedding, d_logits)

        # Gradient for embedding (this flows back to W_embed!)
        d_embedding = self.W_output @ d_logits

        # The gradient for W_embed[center_idx] IS d_embedding
        # Only the row for the center word gets updated
        old_embed = self.W_embed[center_idx].copy()

        # Update
        self.W_output -= lr * d_W_output
        self.W_embed[center_idx] -= lr * d_embedding

        new_embed = self.W_embed[center_idx].copy()

        return {
            'center_word': I2W[center_idx],
            'context_word': I2W[context_idx],
            'loss': round(float(loss), 4),
            'old_embedding': [round(float(v), 4) for v in old_embed],
            'gradient': [round(float(v), 4) for v in d_embedding],
            'new_embedding': [round(float(v), 4) for v in new_embed],
            'embedding_change': [round(float(new_embed[i] - old_embed[i]), 4) for i in range(len(old_embed))],
            'predicted_probs': {I2W[i]: round(float(probs[i]), 4) for i in range(self.vocab_size)},
        }

    def get_all_embeddings(self):
        return {I2W[i]: [round(float(v), 4) for v in self.W_embed[i]] for i in range(self.vocab_size)}

    def train_multiple(self, pairs, lr=0.1, epochs=1):
        """Train on pairs and return embedding snapshots."""
        snapshots = [{'epoch': 0, 'embeddings': self.get_all_embeddings()}]
        step_details = []

        for ep in range(epochs):
            np.random.shuffle(pairs)
            for center, context in pairs:
                detail = self.train_step_detailed(center, context, lr)
                step_details.append(detail)
            snapshots.append({'epoch': ep + 1, 'embeddings': self.get_all_embeddings()})

        return {'snapshots': snapshots, 'steps': step_details[:20]}  # first 20 steps


def demo_forward(word, seed=42):
    """Show the full forward pass for a single word."""
    net = Word2VecNetwork(V, embed_dim=2, seed=seed)
    idx = W2I.get(word, 0)
    result = net.forward_detailed(idx)
    result['word'] = word
    result['vocab'] = VOCAB
    result['embed_dim'] = 2
    result['W_embed_full'] = [[round(float(v), 4) for v in row] for row in net.W_embed]
    return result


def demo_training(num_epochs=30, lr=0.15, seed=42):
    """Train Word2Vec and show how embeddings evolve."""
    # Training pairs from "the cat sat on the mat . the dog sat on the mat ."
    text = "the cat sat on the mat the dog sat on the mat the big cat ran the big dog ran"
    words = text.split()
    pairs = []
    for i in range(len(words)):
        center = W2I.get(words[i])
        if center is None:
            continue
        for j in range(max(0, i-2), min(len(words), i+3)):
            if j != i:
                context = W2I.get(words[j])
                if context is not None:
                    pairs.append([center, context])

    net = Word2VecNetwork(V, embed_dim=2, seed=seed)
    result = net.train_multiple(pairs, lr=lr, epochs=num_epochs)
    result['vocab'] = VOCAB
    result['num_pairs'] = len(pairs)
    result['embed_dim'] = 2
    return result


def real_world_comparison():
    """Info about real-world embedding systems."""
    return {
        'systems': [
            {
                'name': 'Word2Vec (2013)',
                'vocab_size': 3000000,
                'embed_dim': 300,
                'total_params': 3000000 * 300,
                'training': 'Predict context words from center word',
                'usage': 'Standalone — train once, use vectors as features in other models',
                'note': 'Pioneered the idea. Trained on Google News (100B words).',
            },
            {
                'name': 'GloVe (2014)',
                'vocab_size': 2200000,
                'embed_dim': 300,
                'total_params': 2200000 * 300,
                'training': 'Factorize word co-occurrence matrix',
                'usage': 'Standalone — similar to Word2Vec but different math',
                'note': 'Stanford. Uses global statistics instead of local windows.',
            },
            {
                'name': 'GPT-2 (2019)',
                'vocab_size': 50257,
                'embed_dim': 768,
                'total_params': 50257 * 768,
                'training': 'Predict next token — embedding trained end-to-end with transformer',
                'usage': 'Part of the model — embedding is just the first layer',
                'note': 'The embedding table is 38.6M params. The full model is 1.5B params.',
            },
            {
                'name': 'GPT-4 (2023, estimated)',
                'vocab_size': 100000,
                'embed_dim': 12288,
                'total_params': 100000 * 12288,
                'training': 'Predict next token — embedding trained end-to-end',
                'usage': 'Part of the model — just the lookup table at the front',
                'note': 'Estimated. Embedding is ~1.2B params. Full model is ~1.8 trillion.',
            },
        ],
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  HOW EMBEDDINGS ACTUALLY WORK")
    print("=" * 55)

    r = demo_forward('cat')
    print(f"\n  Forward pass for 'cat':")
    print(f"    One-hot: {r['one_hot']}")
    print(f"    Embedding (row select): {r['embedding']}")
    print(f"    Predictions: { {k:v for k,v in sorted(zip(VOCAB, r['probs']), key=lambda x:-x[1])[:3]} }")

    r = demo_training(num_epochs=20)
    print(f"\n  After 20 epochs on {r['num_pairs']} pairs:")
    final = r['snapshots'][-1]['embeddings']
    for w in ['cat', 'dog', 'sat', 'ran', 'the', 'mat']:
        print(f"    {w:4s}: [{final[w][0]:+.3f}, {final[w][1]:+.3f}]")
