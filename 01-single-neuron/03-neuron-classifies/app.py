"""
Module 01.3: A Neuron Classifies - Interactive Demo

Run from the main hub:  python app.py  (from project root)
Or standalone:          python app.py  (from this directory, port 5003)
"""

from flask import Flask, render_template, jsonify, request
from nn_code import ClassifierNeuron, make_dataset, DATASETS

app = Flask(__name__)
neuron = ClassifierNeuron()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    """Set neuron params, classify the grid + points, return everything."""
    data = request.json

    neuron.set_params(data['w1'], data['w2'], data['bias'])

    result = {
        'grid': neuron.predict_grid(),
        'boundary': neuron.get_decision_boundary(),
    }

    # If points are provided, compute accuracy
    if 'points' in data and data['points']:
        result['accuracy'] = neuron.compute_accuracy(data['points'])

    return jsonify(result)


@app.route('/api/dataset/<name>')
def dataset(name):
    """Return a named sample dataset."""
    if name not in DATASETS:
        return jsonify({'error': 'Unknown dataset'}), 404
    points = make_dataset(name)
    return jsonify({'name': name, 'description': DATASETS[name], 'points': points})


@app.route('/api/datasets')
def list_datasets():
    """List available datasets."""
    return jsonify(DATASETS)


if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 01.3: A Neuron Classifies        |")
    print("  |  Open http://localhost:5003 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5003)
