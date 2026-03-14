"""Module 04: Build & Train an MLP — Interactive Playground"""
from flask import Flask, render_template, jsonify, request
import numpy as np
from nn_code import MLP, make_dataset, DATASETS

app = Flask(__name__)
mlp = None
current_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset/<name>')
def dataset(name):
    global current_data
    if name not in DATASETS:
        return jsonify({'error': 'Unknown'}), 404
    ds = make_dataset(name)
    current_data = {'X': np.array(ds['X']), 'labels': np.array(ds['labels']).reshape(-1, 1)}
    return jsonify({'points': ds['points'], 'desc': DATASETS[name]})

@app.route('/api/build', methods=['POST'])
def build():
    global mlp
    data = request.json
    sizes = [2] + data['hidden_layers'] + [1]
    mlp = MLP(sizes, activation=data.get('activation', 'relu'), seed=42)
    result = {'info': mlp.get_info()}
    if current_data is not None:
        result['heatmap'] = mlp.predict_grid()
    return jsonify(result)

@app.route('/api/train', methods=['POST'])
def train():
    if mlp is None or current_data is None:
        return jsonify({'error': 'Build network and load data first'}), 400
    data = request.json
    lr = float(data.get('lr', 0.5))
    steps = int(data.get('steps', 50))
    results = mlp.train_many(current_data['X'], current_data['labels'], lr, steps)
    return jsonify({
        'steps': results,
        'heatmap': mlp.predict_grid(),
        'info': mlp.get_info(),
        'history': mlp.train_history,
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    global mlp
    if mlp is not None:
        sizes = mlp.layer_sizes
        act = mlp.activation
        mlp = MLP(sizes, activation=act, seed=42)
        result = {'info': mlp.get_info()}
        if current_data is not None:
            result['heatmap'] = mlp.predict_grid()
        return jsonify(result)
    return jsonify({})

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 04: MLP Playground               |")
    print("  |  Open http://localhost:5011 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5011)
