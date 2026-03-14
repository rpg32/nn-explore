# Neural Networks: From Zero to Transformers
## A Hands-On, Interactive Learning Journey

---

## Philosophy

- **No black boxes.** Every neural network is built from scratch in Python. You see every weight, every gradient, every activation.
- **One concept at a time.** Each module isolates a single idea so it clicks before moving on.
- **Interactive apps.** Every concept has a standalone app — run `python app.py`, open your browser. Drag sliders, click buttons, watch neurons fire.
- **Build up, never skip.** Later concepts use earlier ones. By the time you reach Transformers, you've built every piece yourself.
- **Two phases of implementation.** Modules 1–5 use **numpy from scratch** so you understand the math. Modules 6–11 introduce **PyTorch + CUDA** so you see how real ML is done — on your RTX 3090.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **NN Code** | Python + NumPy (Modules 1–5) | Build everything from scratch — every forward pass, every gradient |
| **NN Code** | Python + PyTorch + CUDA (Modules 6–11) | Real GPU-accelerated training on your RTX 3090 |
| **Backend** | Flask | Lightweight web server bridging Python computation to the browser |
| **Frontend** | HTML + JavaScript + Canvas | Interactive visualizations, controls, step-through animations |
| **Communication** | REST API (JSON) | JS frontend calls Python backend for computation |

## How Each Module Works

```
module-folder/
├── app.py              ← Run this! Starts Flask server, open browser
├── nn_code.py          ← The actual NN implementation (the learning material)
├── static/
│   └── style.css       ← Shared styling
└── templates/
    └── index.html      ← Interactive frontend (JS + Canvas)
```

**Workflow for each concept:**
1. **Read `nn_code.py`** — This is the core lesson. Pure Python, heavily commented.
2. **Run `python app.py`** — Starts the interactive demo in your browser.
3. **Play with the app** — Drag sliders, add data, step through computations.
4. **Modify `nn_code.py`** — Experiment! Change things, break things, learn.

---

## The Learning Path

### Phase 1: The Building Blocks

> *"What is a neuron, and how does it learn?"*
> **Implementation: NumPy from scratch — you write every equation.**

#### Module 01 — The Single Neuron
The fundamental unit of every neural network.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 1.1 | **What is a Neuron?** | Inputs, weights, bias, weighted sum — the simplest possible computation | Interactive neuron: drag weight/bias sliders, see output change in real-time |
| 1.2 | **Activation Functions** | Why linear isn't enough. Step, Sigmoid, Tanh, ReLU — what each does and why | Side-by-side function grapher: see how each activation shapes the output |
| 1.3 | **A Neuron Classifies** | Using a single neuron to draw a decision boundary on 2D data | 2D scatter plot with a draggable decision line, see classification happen |

#### Module 02 — How Learning Works
The engine that powers every neural network.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 2.1 | **Loss Functions** | How to measure "how wrong" a prediction is (MSE, Cross-Entropy) | Feed in predictions, see the loss value change, understand the landscape |
| 2.2 | **Gradient Descent** | "Roll downhill" — the core optimization algorithm | 2D/3D loss landscape: watch a ball roll to the minimum |
| 2.3 | **Learning Rate** | Too fast (overshoots), too slow (stuck), just right | Same landscape with a learning rate slider — see the difference |
| 2.4 | **A Neuron Learns** | Putting it together: a single neuron training on data, step by step | Watch a neuron adjust weights over epochs, decision boundary moves |

#### Module 03 — Backpropagation
How gradients flow backward through a network.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 3.1 | **The Chain Rule** | Simple calculus: if y = f(g(x)), how does x affect y? | Visual chain of functions — drag x, see effects propagate |
| 3.2 | **Backprop Step-by-Step** | Following gradients backward through a tiny network | Step-through animator: forward pass → compute loss → backward pass, see every gradient |
| 3.3 | **Gradient Flow** | Why gradients can vanish or explode, and what that means | Network diagram with gradient magnitude coloring |

---

### Phase 2: Real Networks

> *"How do layers of neurons work together?"*
> **Implementation: Still NumPy from scratch — now with matrix operations.**

#### Module 04 — Multi-Layer Networks (MLPs)
Stacking neurons into layers creates real power.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 4.1 | **Layers** | Input, hidden, output — how data flows through a stack | Network diagram: see activations light up layer by layer |
| 4.2 | **The Forward Pass** | Computing output from input, one layer at a time | Step-through forward pass with matrix math shown |
| 4.3 | **Universal Approximation** | Why hidden layers can learn *any* function (with enough neurons) | Add neurons to a hidden layer, watch the decision boundary get more complex |
| 4.4 | **Build & Train an MLP** | Your first real network: classify 2D data with hidden layers | **Playground**: choose layers, neurons, activation, dataset — train and watch |

#### Module 05 — Training in Practice
Making training actually work well.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 5.1 | **Batches & Epochs** | Why we don't train on all data at once (mini-batch SGD) | Compare full-batch, mini-batch, and stochastic — see the tradeoffs |
| 5.2 | **Overfitting** | When the network memorizes instead of learning | Train/test split: watch training loss drop while test loss rises |
| 5.3 | **Regularization** | Dropout, L2 penalty — keeping the network honest | Toggle regularization on/off, see the effect on overfitting |
| 5.4 | **Optimizers** | SGD, Momentum, Adam — smarter ways to descend | Side-by-side optimizer race on the same loss landscape |

---

### Phase 3: Specialized Architectures

> *"How do we handle images and sequences?"*
> **Implementation: PyTorch + CUDA — real GPU-accelerated training on your RTX 3090.**
> *(We show the from-scratch NumPy version side-by-side with PyTorch so you see what the framework does for you.)*

#### Module 06 — Convolutional Neural Networks (CNNs)
Learning to see patterns in images.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 6.1 | **The Convolution Operation** | Sliding a small filter over an image — what it computes | Draw on a canvas, apply filters (edge detection, blur, sharpen) live |
| 6.2 | **Feature Maps & Filters** | Multiple filters = multiple feature maps. What the network "sees" | Load an image, visualize what each learned filter detects |
| 6.3 | **Pooling & Stride** | Shrinking the representation — keeping what matters | Interactive pooling: see max-pool and average-pool reduce an image |
| 6.4 | **Build a CNN** | Stack conv → relu → pool layers to classify simple images | Draw digits, CNN classifies them — see feature maps at each layer |

#### Module 07 — Recurrent Neural Networks (RNNs)
Learning from sequences — text, time series, music.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 7.1 | **Why Sequences Are Special** | Order matters. "Dog bites man" ≠ "Man bites dog" | Feed the same words in different orders, see different outputs |
| 7.2 | **Recurrent Connections** | The hidden state — a network with memory | Step through a sequence, watch the hidden state evolve |
| 7.3 | **Vanishing Gradients** | Why vanilla RNNs forget long-ago inputs | Gradient flow visualization through time steps — watch it shrink |
| 7.4 | **LSTM: Long Short-Term Memory** | Gates (forget, input, output) that control what to remember | LSTM cell diagram: see gates open/close as a sequence flows through |
| 7.5 | **Build a Text Predictor** | Character-level RNN/LSTM that learns to predict the next letter | Type text, train the model, see it generate continuations |

---

### Phase 4: Attention & Transformers

> *"The architecture that changed everything."*
> **Implementation: PyTorch + CUDA — with every component also explained from first principles.**

#### Module 08 — Embeddings & Representation
Before attention, we need to understand how words become numbers.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 8.1 | **One-Hot Encoding** | The naive approach and why it falls short | Visualize one-hot vectors — see that "cat" and "dog" are equally far apart |
| 8.2 | **Dense Embeddings** | Meaning as geometry — similar words are nearby | 2D embedding space: drag words around, see relationships |
| 8.3 | **Learning Embeddings** | How embeddings are trained (Word2Vec skip-gram intuition) | Train tiny embeddings on a small corpus, watch words cluster by meaning |

#### Module 09 — The Attention Mechanism
The key insight: not all inputs are equally important.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 9.1 | **The Bottleneck Problem** | Why compressing a whole sentence into one vector loses information | RNN encoder-decoder: watch information get squeezed and lost |
| 9.2 | **Attention as Weighted Lookup** | Query, Key, Value — the three vectors that make attention work | Interactive Q/K/V: see how a query "matches" different keys |
| 9.3 | **Attention Scores & Weights** | Dot product → softmax → weighted sum, step by step | Type a sentence, see the attention weight matrix form |
| 9.4 | **Self-Attention** | A sequence attending to itself — every word looks at every other word | **The big one**: type text, see the self-attention pattern light up |

#### Module 10 — The Transformer
Putting all the pieces together.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 10.1 | **Positional Encoding** | No recurrence = no order. How transformers know "where" each token is | Visualize sinusoidal position encodings as colored patterns |
| 10.2 | **Multi-Head Attention** | Multiple attention "perspectives" in parallel | See how different heads attend to different relationships |
| 10.3 | **The Transformer Block** | Attention → Add & Norm → Feed-Forward → Add & Norm | Step through one block, see data transform at each stage |
| 10.4 | **Encoder vs. Decoder** | Two variants: understanding vs. generating | Side-by-side architecture diagrams with data flow |
| 10.5 | **Masked Attention** | Why decoders can't peek at the future | Attention matrix with mask — see the triangular pattern |
| 10.6 | **The Full Architecture** | The complete Transformer, end to end | **Capstone**: interactive full transformer diagram with data flow |

#### Module 11 — Building a Tiny Language Model
The grand finale: you build a small GPT-like model.

| # | Concept | What You'll Learn | App |
|---|---------|-------------------|-----|
| 11.1 | **Tokenization** | How text becomes numbers (character-level, BPE intuition) | Tokenizer playground: type text, see tokens |
| 11.2 | **Training a Tiny GPT** | Putting together embeddings + transformer blocks + training loop | Train on a small text corpus on your GPU, watch loss decrease |
| 11.3 | **Generation** | Sampling, temperature, top-k — how text is produced | Generate text with temperature/top-k sliders, see probability distributions |
| 11.4 | **What Did It Learn?** | Probing the trained model — attention patterns, embeddings | Inspect your trained model's internals |

---

## Project Structure

```
nn-explore/
├── PLAN.md                         ← This file (your roadmap)
├── requirements.txt                ← Python dependencies
├── shared/
│   ├── nn_utils.py                 ← Shared math utilities (built up as we go)
│   └── static/
│       └── common.css              ← Shared styling for all apps
│
├── 01-single-neuron/
│   ├── 01-what-is-a-neuron/
│   │   ├── app.py                  ← Flask server (run this)
│   │   ├── nn_code.py              ← Neuron implementation from scratch
│   │   └── templates/
│   │       └── index.html          ← Interactive frontend
│   ├── 02-activation-functions/
│   │   ├── app.py
│   │   ├── nn_code.py
│   │   └── templates/
│   │       └── index.html
│   └── 03-neuron-classifies/
│       ├── app.py
│       ├── nn_code.py
│       └── templates/
│           └── index.html
│
├── 02-learning/
│   ├── 01-loss-functions/
│   ├── 02-gradient-descent/
│   ├── 03-learning-rate/
│   └── 04-neuron-learns/
│
│   ... (same pattern for all modules)
│
└── 11-tiny-language-model/
    ├── 01-tokenization/
    ├── 02-training-tiny-gpt/
    ├── 03-generation/
    └── 04-what-did-it-learn/
```

## Suggested Pace

- **Modules 1–3** (Foundations): ~1 week — Take your time here, these are the bedrock
- **Modules 4–5** (Real Networks): ~1 week — This is where it starts feeling "real"
- **Modules 6–7** (CNNs & RNNs): ~1 week — Specialized architectures, now GPU-powered
- **Modules 8–9** (Embeddings & Attention): ~1 week — The crucial bridge to transformers
- **Modules 10–11** (Transformers & Mini-GPT): ~1–2 weeks — The grand finale

Total: **5–6 weeks** at a comfortable pace, or faster if you're hooked.

## Prerequisites

- Python 3.11 ✓
- NumPy ✓
- PyTorch + CUDA ✓ (RTX 3090)
- Flask ✓
- A web browser
- Curiosity
