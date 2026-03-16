"""Module 07.5: Build a Text Predictor — Character-Level LSTM"""
from flask import Flask, render_template, jsonify, request
from nn_code import TextPredictor, get_corpus_info, TRAINING_TEXT

app = Flask(__name__)

# Global model instance
model = TextPredictor(hidden_size=64, seed=42)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Return current model state and corpus info."""
    return jsonify({
        'model': model.get_state(),
        'corpus': get_corpus_info(),
    })


@app.route('/api/train', methods=['POST'])
def train():
    """Train the model for N epochs.

    Expects JSON: { epochs: int, lr: float }
    Returns: { losses: [...], model: {...} }
    """
    data = request.json
    epochs = min(int(data.get('epochs', 5)), 50)  # Cap at 50 per request
    lr = float(data.get('lr', 0.01))

    losses = []
    for _ in range(epochs):
        loss = model.train_epoch(lr=lr)
        losses.append(loss)

    return jsonify({
        'losses': losses,
        'model': model.get_state(),
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text from a seed string.

    Expects JSON: { seed: str, length: int, temperature: float }
    Returns: { text: str, seed: str, temperature: float }
    """
    data = request.json
    seed = data.get('seed', 'the ')
    length = min(int(data.get('length', 100)), 300)
    temperature = float(data.get('temperature', 0.8))

    text = model.generate(seed_text=seed, length=length, temperature=temperature)

    return jsonify({
        'text': text,
        'seed': seed,
        'length': length,
        'temperature': temperature,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Get predictions at each position in the text.

    Expects JSON: { text: str }
    Returns: { predictions: [...] }
    """
    data = request.json
    text = data.get('text', 'the cat')

    predictions = model.get_predictions(text)

    return jsonify({
        'predictions': predictions,
        'text': text,
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the model to untrained state."""
    global model
    model = TextPredictor(hidden_size=64, seed=42)
    return jsonify({
        'model': model.get_state(),
        'message': 'Model reset to untrained state',
    })


if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 07.5: Text Predictor             |")
    print("  |  Open http://localhost:5026 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5026)
