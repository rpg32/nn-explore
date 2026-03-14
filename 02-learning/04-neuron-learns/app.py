"""Module 02.4: A Neuron Learns - Interactive Demo"""

from flask import Flask, render_template, jsonify, request
import numpy as np
from nn_code import TrainableNeuron, make_dataset, DATASETS

app = Flask(__name__)
neuron = TrainableNeuron()
current_data = None  # holds X, labels as numpy arrays

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset/<name>')
def dataset(name):
    global current_data
    if name not in DATASETS:
        return jsonify({'error': 'Unknown'}), 404
    ds = make_dataset(name)
    current_data = {'X': np.array(ds['X']), 'labels': np.array(ds['labels'])}
    neuron.reset()
    return jsonify({'points': ds['points'], 'desc': DATASETS[name],
                    'state': neuron.get_state()})

@app.route('/api/step', methods=['POST'])
def step():
    if current_data is None:
        return jsonify({'error': 'Load a dataset first'}), 400
    data = request.json
    lr = float(data.get('lr', 1.0))
    result = neuron.train_step(current_data['X'], current_data['labels'], lr)
    result['boundary'] = neuron.get_decision_boundary()
    result['heatmap'] = neuron.get_heatmap()
    result['state'] = neuron.get_state()
    return jsonify(result)

@app.route('/api/run', methods=['POST'])
def run():
    if current_data is None:
        return jsonify({'error': 'Load a dataset first'}), 400
    data = request.json
    lr = float(data.get('lr', 1.0))
    steps = int(data.get('steps', 50))
    results = neuron.train_many(current_data['X'], current_data['labels'], lr, steps)
    return jsonify({
        'steps': results,
        'boundary': neuron.get_decision_boundary(),
        'heatmap': neuron.get_heatmap(),
        'state': neuron.get_state(),
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    neuron.reset()
    result = {'state': neuron.get_state()}
    result['boundary'] = neuron.get_decision_boundary()
    result['heatmap'] = neuron.get_heatmap()
    return jsonify(result)

@app.route('/api/state')
def state():
    result = neuron.get_state()
    result['boundary'] = neuron.get_decision_boundary()
    result['heatmap'] = neuron.get_heatmap()
    result['history'] = neuron.history
    return jsonify(result)

if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 02.4: A Neuron Learns            |")
    print("  |  Open http://localhost:5007 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5007)
