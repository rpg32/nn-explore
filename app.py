"""
Neural Networks: From Zero to Transformers
==========================================
Main hub — run this once, access all lessons in your browser.

    python app.py

Then open http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, send_file
import importlib.util
import os

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


# =============================================================
# MODULE REGISTRY — add new lessons here as we build them
# =============================================================
MODULES = [
    {
        'id': '01-1',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '01',
        'group_name': 'The Single Neuron',
        'title': 'What is a Neuron?',
        'desc': 'Inputs, weights, bias, weighted sum — the simplest possible computation.',
        'path': '01-single-neuron/01-what-is-a-neuron',
        'status': 'ready',
    },
    {
        'id': '01-2',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '01',
        'group_name': 'The Single Neuron',
        'title': 'Activation Functions',
        'desc': 'Why nonlinearity matters — Step, Sigmoid, Tanh, ReLU visualized.',
        'path': '01-single-neuron/02-activation-functions',
        'status': 'ready',
    },
    {
        'id': '01-3',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '01',
        'group_name': 'The Single Neuron',
        'title': 'A Neuron Classifies',
        'desc': 'Using a single neuron to draw a decision boundary on 2D data.',
        'path': '01-single-neuron/03-neuron-classifies',
        'status': 'ready',
    },
    {
        'id': '02-1',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '02',
        'group_name': 'How Learning Works',
        'title': 'Loss Functions',
        'desc': 'How to measure "how wrong" a prediction is — MSE vs Cross-Entropy.',
        'path': '02-learning/01-loss-functions',
        'status': 'ready',
    },
    {
        'id': '02-2',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '02',
        'group_name': 'How Learning Works',
        'title': 'Gradient Descent',
        'desc': 'The core optimization algorithm — rolling downhill on the loss landscape.',
        'path': '02-learning/02-gradient-descent',
        'status': 'ready',
    },
    {
        'id': '02-3',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '02',
        'group_name': 'How Learning Works',
        'title': 'Learning Rate',
        'desc': 'Too slow, too fast, just right — three runners race on the same landscape.',
        'path': '02-learning/03-learning-rate',
        'status': 'ready',
    },
    {
        'id': '02-4',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '02',
        'group_name': 'How Learning Works',
        'title': 'A Neuron Learns',
        'desc': 'Everything together — watch a neuron train itself on data, step by step.',
        'path': '02-learning/04-neuron-learns',
        'status': 'ready',
    },
    {
        'id': '03-1',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '03',
        'group_name': 'Backpropagation',
        'title': 'The Chain Rule',
        'desc': 'How gradients flow backward through a computation graph — the key to deep learning.',
        'path': '03-backpropagation/01-chain-rule',
        'status': 'ready',
    },
    {
        'id': '03-2',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '03',
        'group_name': 'Backpropagation',
        'title': 'Backprop Step-by-Step',
        'desc': 'Watch gradients flow backward through a 2-layer network — forward, backward, update.',
        'path': '03-backpropagation/02-backprop-step-by-step',
        'status': 'ready',
    },
    {
        'id': '03-3',
        'phase': 1,
        'phase_name': 'The Building Blocks',
        'group': '03',
        'group_name': 'Backpropagation',
        'title': 'Gradient Flow',
        'desc': 'Why gradients vanish in deep networks — and how ReLU fixes it.',
        'path': '03-backpropagation/03-gradient-flow',
        'status': 'ready',
    },
    {
        'id': '04-1',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '04',
        'group_name': 'Multi-Layer Networks',
        'title': 'Layers',
        'desc': 'Input, hidden, and output — how neurons are organized into layers.',
        'path': '04-multi-layer-networks/01-layers',
        'status': 'ready',
    },
    {
        'id': '04-2',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '04',
        'group_name': 'Multi-Layer Networks',
        'title': 'The Forward Pass',
        'desc': 'Step through the matrix math — see exactly how data transforms at each layer.',
        'path': '04-multi-layer-networks/02-forward-pass',
        'status': 'ready',
    },
    {
        'id': '04-3',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '04',
        'group_name': 'Multi-Layer Networks',
        'title': 'Universal Approximation',
        'desc': 'More neurons = finer fit — like refining a mesh. One hidden layer can learn any function.',
        'path': '04-multi-layer-networks/03-universal-approximation',
        'status': 'ready',
    },
    {
        'id': '04-4',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '04',
        'group_name': 'Multi-Layer Networks',
        'title': 'MLP Playground',
        'desc': 'Build a network, pick a dataset, train it, watch the decision boundary form.',
        'path': '04-multi-layer-networks/01-build-train-mlp',
        'status': 'ready',
    },
    {
        'id': '05-1',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '05',
        'group_name': 'Training Techniques',
        'title': 'Batch vs SGD',
        'desc': 'Same network, same data, three ways to feed gradients — watch the difference.',
        'path': '05-training-techniques/01-batch-vs-sgd',
        'status': 'ready',
    },
    {
        'id': '05-2',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '05',
        'group_name': 'Training Techniques',
        'title': 'Optimizers',
        'desc': 'Why plain gradient descent is slow — and how Momentum, RMSProp, and Adam fix it.',
        'path': '05-training-techniques/02-optimizers',
        'status': 'ready',
    },
    {
        'id': '05-3',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '05',
        'group_name': 'Training Techniques',
        'title': 'Overfitting & Regularization',
        'desc': 'When your network memorizes noise — and how L2 and Dropout fix it.',
        'path': '05-training-techniques/03-regularization',
        'status': 'ready',
    },
    {
        'id': '05-4',
        'phase': 2,
        'phase_name': 'Real Networks',
        'group': '05',
        'group_name': 'Training Techniques',
        'title': 'Data Normalization',
        'desc': 'Why raw data breaks training and how scaling your inputs fixes it.',
        'path': '05-training-techniques/04-normalization',
        'status': 'ready',
    },
    {
        'id': '06-1',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '06',
        'group_name': 'CNNs',
        'title': 'Convolution',
        'desc': 'A small filter slides over an image, computing weighted sums — detecting features everywhere.',
        'path': '06-cnns/01-convolution',
        'status': 'ready',
    },
    {
        'id': '06-2',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '06',
        'group_name': 'CNNs',
        'title': 'Feature Maps & Filters',
        'desc': 'Multiple kernels applied simultaneously — each one detects a different feature in the same image.',
        'path': '06-cnns/02-feature-maps',
        'status': 'ready',
    },
    {
        'id': '06-3',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '06',
        'group_name': 'CNNs',
        'title': 'Pooling & Stride',
        'desc': 'The full pipeline — convolution, pooling, stacking, flattening to classification.',
        'path': '06-cnns/02-building-a-cnn',
        'status': 'ready',
    },
    {
        'id': '06-4',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '06',
        'group_name': 'CNNs',
        'title': 'Build & Train a CNN',
        'desc': 'Train a CNN on handwritten digits — watch kernels learn to detect features.',
        'path': '06-cnns/04-build-train-cnn',
        'status': 'coming',
    },
    {
        'id': '07-1',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '07',
        'group_name': 'RNNs',
        'title': 'Why Sequences Are Special',
        'desc': 'Order matters — why MLPs fail at language and time series.',
        'path': '07-rnns/01-why-sequences',
        'status': 'ready',
    },
    {
        'id': '07-2',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '07',
        'group_name': 'RNNs',
        'title': 'Recurrent Connections',
        'desc': 'Processing sequences one element at a time, carrying memory in a hidden state.',
        'path': '07-rnns/01-recurrence',
        'status': 'ready',
    },
    {
        'id': '07-4',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '07',
        'group_name': 'RNNs',
        'title': 'LSTM',
        'desc': 'Long Short-Term Memory — gates and a cell state highway that fix vanishing gradients.',
        'path': '07-rnns/04-lstm',
        'status': 'ready',
    },
    {
        'id': '07-5',
        'phase': 3,
        'phase_name': 'Deep Learning',
        'group': '07',
        'group_name': 'RNNs',
        'title': 'Build a Text Predictor',
        'desc': 'Train a character-level LSTM — watch it learn to predict English from scratch.',
        'path': '07-rnns/05-text-predictor',
        'status': 'ready',
    },
    {
        'id': '08-1',
        'phase': 4,
        'phase_name': 'Attention & Transformers',
        'group': '08',
        'group_name': 'Embeddings',
        'title': 'One-Hot Encoding',
        'desc': 'The simplest way to turn words into numbers — and why it\'s not enough.',
        'path': '08-embeddings/01-one-hot',
        'status': 'ready',
    },
]


# =============================================================
# DYNAMIC MODULE LOADING
# =============================================================
def load_nn_code(name, subpath):
    """Load a module's nn_code.py with a unique name to avoid conflicts."""
    filepath = os.path.join(BASE, subpath, 'nn_code.py')
    if not os.path.exists(filepath):
        return None
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load all ready modules
nn = {}
for m in MODULES:
    if m['status'] == 'ready':
        key = m['id']
        nn[key] = load_nn_code(f"nn_{key.replace('-', '_')}", m['path'])


# =============================================================
# HUB PAGE
# =============================================================
@app.route('/')
def hub():
    return render_template('hub.html', modules=MODULES)


# =============================================================
# MODULE 01-1: What is a Neuron?
# =============================================================
_neuron_01_1 = nn['01-1'].Neuron(num_inputs=3)

@app.route('/01-1/')
def m01_1_page():
    return send_file(os.path.join(BASE, '01-single-neuron/01-what-is-a-neuron/templates/index.html'))

@app.route('/01-1/api/compute', methods=['POST'])
def m01_1_compute():
    data = request.json
    _neuron_01_1.set_weights(data['weights'])
    _neuron_01_1.set_bias(data['bias'])
    return jsonify(_neuron_01_1.forward(data['inputs']))


# =============================================================
# MODULE 01-2: Activation Functions
# =============================================================
@app.route('/01-2/')
def m01_2_page():
    return send_file(os.path.join(BASE, '01-single-neuron/02-activation-functions/templates/index.html'))

@app.route('/01-2/api/activate', methods=['POST'])
def m01_2_activate():
    data = request.json
    x = float(data['x'])
    results = {}
    for name in nn['01-2'].ACTIVATIONS:
        results[name] = nn['01-2'].compute_activation(name, x)
    return jsonify(results)

@app.route('/01-2/api/curves')
def m01_2_curves():
    result = {}
    for name in nn['01-2'].ACTIVATIONS:
        result[name] = nn['01-2'].compute_curve(name)
    return jsonify(result)

@app.route('/01-2/api/linearity')
def m01_2_linearity():
    return jsonify(nn['01-2'].demonstrate_linearity_problem())


# =============================================================
# MODULE 01-3: A Neuron Classifies
# =============================================================
_neuron_01_3 = nn['01-3'].ClassifierNeuron()

@app.route('/01-3/')
def m01_3_page():
    return send_file(os.path.join(BASE, '01-single-neuron/03-neuron-classifies/templates/index.html'))

@app.route('/01-3/api/classify', methods=['POST'])
def m01_3_classify():
    data = request.json
    _neuron_01_3.set_params(data['w1'], data['w2'], data['bias'])
    result = {
        'grid': _neuron_01_3.predict_grid(),
        'boundary': _neuron_01_3.get_decision_boundary(),
    }
    if 'points' in data and data['points']:
        result['accuracy'] = _neuron_01_3.compute_accuracy(data['points'])
    return jsonify(result)

@app.route('/01-3/api/dataset/<name>')
def m01_3_dataset(name):
    if name not in nn['01-3'].DATASETS:
        return jsonify({'error': 'Unknown dataset'}), 404
    return jsonify({
        'name': name,
        'description': nn['01-3'].DATASETS[name],
        'points': nn['01-3'].make_dataset(name),
    })


# =============================================================
# MODULE 02-1: Loss Functions
# =============================================================
@app.route('/02-1/')
def m02_1_page():
    return send_file(os.path.join(BASE, '02-learning/01-loss-functions/templates/index.html'))

@app.route('/02-1/api/compute', methods=['POST'])
def m02_1_compute():
    data = request.json
    y_true = float(data['y_true'])
    y_pred = float(data['y_pred'])
    result = nn['02-1'].compute_both(y_true, y_pred)
    result['mse_curve'] = nn['02-1'].mse_curve(y_true)
    result['bce_curve'] = nn['02-1'].bce_curve(y_true)
    return jsonify(result)

@app.route('/02-1/api/batch', methods=['POST'])
def m02_1_batch():
    data = request.json
    return jsonify(nn['02-1'].compute_batch_loss(data['predictions'], data.get('loss_type', 'bce')))


# =============================================================
# MODULE 02-2: Gradient Descent
# =============================================================
@app.route('/02-2/')
def m02_2_page():
    return send_file(os.path.join(BASE, '02-learning/02-gradient-descent/templates/index.html'))

@app.route('/02-2/api/landscape/<name>')
def m02_2_landscape(name):
    if name not in nn['02-2'].LANDSCAPES:
        return jsonify({'error': 'Unknown'}), 404
    return jsonify(nn['02-2'].compute_landscape(name))

@app.route('/02-2/api/step', methods=['POST'])
def m02_2_step():
    data = request.json
    return jsonify(nn['02-2'].gradient_descent_step(
        data['landscape'], data['w1'], data['w2'], data['lr']))

@app.route('/02-2/api/run', methods=['POST'])
def m02_2_run():
    data = request.json
    return jsonify(nn['02-2'].run_descent(
        data['landscape'], data['w1'], data['w2'],
        data['lr'], data.get('steps', 50)))

@app.route('/02-2/api/landscapes')
def m02_2_landscapes():
    return jsonify({k: {'label': v['label'], 'desc': v['desc']}
                    for k, v in nn['02-2'].LANDSCAPES.items()})


# =============================================================
# MODULE 02-3: Learning Rate
# =============================================================
@app.route('/02-3/')
def m02_3_page():
    return send_file(os.path.join(BASE, '02-learning/03-learning-rate/templates/index.html'))

@app.route('/02-3/api/landscape')
def m02_3_landscape():
    return jsonify(nn['02-3'].compute_landscape())

@app.route('/02-3/api/run', methods=['POST'])
def m02_3_run():
    data = request.json
    return jsonify(nn['02-3'].run_three(
        data['w1'], data['w2'],
        data['lr_slow'], data['lr_good'], data['lr_fast'],
        data.get('steps', 60)))

@app.route('/02-3/api/sweep', methods=['POST'])
def m02_3_sweep():
    data = request.json
    return jsonify(nn['02-3'].lr_sweep(data['w1'], data['w2']))


# =============================================================
# MODULE 02-4: A Neuron Learns
# =============================================================
_neuron_02_4 = nn['02-4'].TrainableNeuron()
_data_02_4 = {}

@app.route('/02-4/')
def m02_4_page():
    return send_file(os.path.join(BASE, '02-learning/04-neuron-learns/templates/index.html'))

@app.route('/02-4/api/dataset/<name>')
def m02_4_dataset(name):
    import numpy as _np
    if name not in nn['02-4'].DATASETS:
        return jsonify({'error': 'Unknown'}), 404
    ds = nn['02-4'].make_dataset(name)
    _data_02_4['X'] = _np.array(ds['X'])
    _data_02_4['labels'] = _np.array(ds['labels'])
    _neuron_02_4.reset()
    return jsonify({'points': ds['points'], 'desc': nn['02-4'].DATASETS[name],
                    'state': _neuron_02_4.get_state()})

@app.route('/02-4/api/step', methods=['POST'])
def m02_4_step():
    if 'X' not in _data_02_4:
        return jsonify({'error': 'Load a dataset first'}), 400
    data = request.json
    result = _neuron_02_4.train_step(_data_02_4['X'], _data_02_4['labels'], float(data.get('lr', 1.0)))
    result['boundary'] = _neuron_02_4.get_decision_boundary()
    result['heatmap'] = _neuron_02_4.get_heatmap()
    result['state'] = _neuron_02_4.get_state()
    return jsonify(result)

@app.route('/02-4/api/run', methods=['POST'])
def m02_4_run():
    if 'X' not in _data_02_4:
        return jsonify({'error': 'Load a dataset first'}), 400
    data = request.json
    results = _neuron_02_4.train_many(_data_02_4['X'], _data_02_4['labels'],
                                       float(data.get('lr', 1.0)), int(data.get('steps', 50)))
    return jsonify({
        'steps': results,
        'boundary': _neuron_02_4.get_decision_boundary(),
        'heatmap': _neuron_02_4.get_heatmap(),
        'state': _neuron_02_4.get_state(),
    })

@app.route('/02-4/api/reset', methods=['POST'])
def m02_4_reset():
    _neuron_02_4.reset()
    return jsonify({
        'state': _neuron_02_4.get_state(),
        'boundary': _neuron_02_4.get_decision_boundary(),
        'heatmap': _neuron_02_4.get_heatmap(),
    })


# =============================================================
# MODULE 03-1: The Chain Rule
# =============================================================
@app.route('/03-1/')
def m03_1_page():
    return send_file(os.path.join(BASE, '03-backpropagation/01-chain-rule/templates/index.html'))

@app.route('/03-1/api/compute', methods=['POST'])
def m03_1_compute():
    data = request.json
    return jsonify(nn['03-1'].compute_graph(
        float(data['x']), float(data['w']),
        float(data['b']), float(data['y_true'])))


# =============================================================
# MODULE 03-2: Backprop Step-by-Step
# =============================================================
_net_03_2 = nn['03-2'].TinyNetwork()

@app.route('/03-2/')
def m03_2_page():
    return send_file(os.path.join(BASE, '03-backpropagation/02-backprop-step-by-step/templates/index.html'))

@app.route('/03-2/api/forward', methods=['POST'])
def m03_2_forward():
    data = request.json
    return jsonify(_net_03_2.forward_detailed(data['x'], float(data['y_true'])))

@app.route('/03-2/api/backward', methods=['POST'])
def m03_2_backward():
    data = request.json
    return jsonify(_net_03_2.backward_detailed(data['x'], float(data['y_true'])))

@app.route('/03-2/api/train_step', methods=['POST'])
def m03_2_train_step():
    data = request.json
    return jsonify(_net_03_2.train_step(data['x'], float(data['y_true']), float(data['lr'])))

@app.route('/03-2/api/train_many', methods=['POST'])
def m03_2_train_many():
    data = request.json
    return jsonify(_net_03_2.train_many(data['x'], float(data['y_true']),
                                         float(data['lr']), int(data.get('steps', 50))))

@app.route('/03-2/api/reset', methods=['POST'])
def m03_2_reset():
    global _net_03_2
    _net_03_2 = nn['03-2'].TinyNetwork()
    return jsonify({'params': _net_03_2.get_params()})


# =============================================================
# MODULE 03-3: Gradient Flow
# =============================================================
@app.route('/03-3/')
def m03_3_page():
    return send_file(os.path.join(BASE, '03-backpropagation/03-gradient-flow/templates/index.html'))

@app.route('/03-3/api/simulate', methods=['POST'])
def m03_3_simulate():
    data = request.json
    return jsonify(nn['03-3'].compare_activations(
        int(data['num_layers']), float(data.get('weight_scale', 1.0))))

@app.route('/03-3/api/sweep', methods=['POST'])
def m03_3_sweep():
    data = request.json
    return jsonify({
        'sigmoid': nn['03-3'].depth_sweep('sigmoid', float(data.get('weight_scale', 1.0))),
        'relu': nn['03-3'].depth_sweep('relu', float(data.get('weight_scale', 1.0))),
    })


# =============================================================
# MODULE 04-1: Layers
# =============================================================
@app.route('/04-1/')
def m04_1_page():
    return send_file(os.path.join(BASE, '04-multi-layer-networks/01-layers/templates/index.html'))

@app.route('/04-1/api/forward', methods=['POST'])
def m04_1_forward():
    data = request.json
    return jsonify(nn['04-1'].demo_forward(
        data['architecture'],
        data.get('inputs', [0.5, -0.3]),
        seed=int(data.get('seed', 42)),
    ))


# =============================================================
# MODULE 04-2: The Forward Pass
# =============================================================
@app.route('/04-2/')
def m04_2_page():
    return send_file(os.path.join(BASE, '04-multi-layer-networks/02-forward-pass/templates/index.html'))

@app.route('/04-2/api/forward', methods=['POST'])
def m04_2_forward():
    data = request.json
    return jsonify(nn['04-2'].run_forward_pass(
        data.get('architecture', [2, 4, 1]),
        data.get('inputs', [0.5, -0.3]),
        seed=int(data.get('seed', 42)),
    ))


# =============================================================
# MODULE 04-3: Universal Approximation
# =============================================================
@app.route('/04-3/')
def m04_3_page():
    return send_file(os.path.join(BASE, '04-multi-layer-networks/03-universal-approximation/templates/index.html'))

@app.route('/04-3/api/fit', methods=['POST'])
def m04_3_fit():
    data = request.json
    max_n = int(data.get('max_neurons', 64))
    neuron_counts = [n for n in [1, 2, 4, 8, 16, 32, 64] if n <= max_n]
    return jsonify(nn['04-3'].fit_with_increasing_neurons(
        data.get('function_name', 'sine'),
        neuron_counts=neuron_counts,
        epochs=int(data.get('epochs', 2000)),
    ))


# =============================================================
# MODULE 04-4: MLP Playground
# =============================================================
import numpy as _np
_mlp_04 = None
_data_04 = {}

@app.route('/04-4/')
def m04_4_page():
    return send_file(os.path.join(BASE, '04-multi-layer-networks/01-build-train-mlp/templates/index.html'))

@app.route('/04-4/api/dataset/<name>')
def m04_4_dataset(name):
    if name not in nn['04-4'].DATASETS:
        return jsonify({'error': 'Unknown'}), 404
    ds = nn['04-4'].make_dataset(name)
    _data_04['X'] = _np.array(ds['X'])
    _data_04['labels'] = _np.array(ds['labels']).reshape(-1, 1)
    return jsonify({'points': ds['points'], 'desc': nn['04-4'].DATASETS[name]})

@app.route('/04-4/api/build', methods=['POST'])
def m04_4_build():
    global _mlp_04
    data = request.json
    sizes = [2] + data['hidden_layers'] + [1]
    _mlp_04 = nn['04-4'].MLP(sizes, activation=data.get('activation', 'relu'), seed=42)
    result = {'info': _mlp_04.get_info()}
    if 'X' in _data_04:
        result['heatmap'] = _mlp_04.predict_grid()
    return jsonify(result)

@app.route('/04-4/api/train', methods=['POST'])
def m04_4_train():
    if _mlp_04 is None or 'X' not in _data_04:
        return jsonify({'error': 'Build network and load data first'}), 400
    data = request.json
    results = _mlp_04.train_many(_data_04['X'], _data_04['labels'],
                                  float(data.get('lr', 0.5)), int(data.get('steps', 50)))
    return jsonify({
        'steps': results,
        'heatmap': _mlp_04.predict_grid(),
        'info': _mlp_04.get_info(),
        'history': _mlp_04.train_history,
    })

@app.route('/04-4/api/reset', methods=['POST'])
def m04_4_reset():
    global _mlp_04
    if _mlp_04:
        _mlp_04 = nn['04-4'].MLP(_mlp_04.layer_sizes, _mlp_04.activation, seed=42)
        result = {'info': _mlp_04.get_info()}
        if 'X' in _data_04:
            result['heatmap'] = _mlp_04.predict_grid()
        return jsonify(result)
    return jsonify({})


# =============================================================
# MODULE 05-1: Batch vs SGD
# =============================================================
@app.route('/05-1/')
def m05_1_page():
    return send_file(os.path.join(BASE, '05-training-techniques/01-batch-vs-sgd/templates/index.html'))

@app.route('/05-1/api/compare', methods=['POST'])
def m05_1_compare():
    data = request.json
    return jsonify(nn['05-1'].train_comparison(
        num_epochs=int(data.get('epochs', 30)),
        lr=float(data.get('lr', 0.5)),
        batch_size=int(data.get('batch_size', 16)),
    ))

@app.route('/05-1/api/traces', methods=['POST'])
def m05_1_traces():
    data = request.json
    return jsonify(nn['05-1'].train_step_by_step(
        total_epochs=int(data.get('epochs', 30)),
        lr=float(data.get('lr', 0.5)),
        batch_size=int(data.get('batch_size', 16)),
    ))


# =============================================================
# MODULE 05-2: Optimizers
# =============================================================
@app.route('/05-2/')
def m05_2_page():
    return send_file(os.path.join(BASE, '05-training-techniques/02-optimizers/templates/index.html'))

@app.route('/05-2/api/landscape', methods=['POST'])
def m05_2_landscape():
    data = request.json
    return jsonify(nn['05-2'].run_optimizers(
        landscape=data.get('landscape', 'bowl'),
        lr=float(data.get('lr', 0.01)),
        steps=int(data.get('steps', 150)),
    ))

@app.route('/05-2/api/nn_compare', methods=['POST'])
def m05_2_nn_compare():
    data = request.json
    return jsonify(nn['05-2'].nn_comparison(
        epochs=int(data.get('epochs', 50)),
        lr=float(data.get('lr', 0.5)),
    ))


# =============================================================
# MODULE 05-3: Overfitting & Regularization
# =============================================================
@app.route('/05-3/')
def m05_3_page():
    return send_file(os.path.join(BASE, '05-training-techniques/03-regularization/templates/index.html'))

@app.route('/05-3/api/compare', methods=['POST'])
def m05_3_compare():
    data = request.json
    return jsonify(nn['05-3'].train_comparison(
        epochs=int(data.get('epochs', 200)),
        lr=float(data.get('lr', 0.5)),
        l2_lambda=float(data.get('l2_lambda', 0.01)),
        dropout_rate=float(data.get('dropout_rate', 0.5)),
    ))


# =============================================================
# MODULE 05-4: Data Normalization
# =============================================================
@app.route('/05-4/')
def m05_4_page():
    return send_file(os.path.join(BASE, '05-training-techniques/04-normalization/templates/index.html'))

@app.route('/05-4/api/compare', methods=['POST'])
def m05_4_compare():
    data = request.json
    return jsonify(nn['05-4'].train_comparison(
        epochs=int(data.get('epochs', 100)),
        lr=float(data.get('lr', 0.5)),
        scale_ratio=float(data.get('scale_ratio', 100)),
    ))


# =============================================================
# MODULE 06-1: Convolution
# =============================================================
@app.route('/06-1/')
def m06_1_page():
    return send_file(os.path.join(BASE, '06-cnns/01-convolution/templates/index.html'))

@app.route('/06-1/api/images')
def m06_1_images():
    return jsonify(nn['06-1'].make_test_images())

@app.route('/06-1/api/kernels')
def m06_1_kernels():
    return jsonify({k: {'name': v['name'], 'desc': v['desc'], 'kernel': v['kernel']}
                    for k, v in nn['06-1'].KERNELS.items()})

@app.route('/06-1/api/apply', methods=['POST'])
def m06_1_apply():
    data = request.json
    return jsonify(nn['06-1'].apply_kernel(data['image'], data['kernel']))

@app.route('/06-1/api/step', methods=['POST'])
def m06_1_step():
    data = request.json
    result = nn['06-1'].compute_single_step(data['image'], data['kernel'], data['row'], data['col'])
    return jsonify(result) if result else ('', 400)


# =============================================================
# MODULE 06-2: Feature Maps & Filters
# =============================================================
_fm_images = nn['06-2'].make_test_images()

@app.route('/06-2/')
def m06_2_page():
    return send_file(os.path.join(BASE, '06-cnns/02-feature-maps/templates/index.html'))

@app.route('/06-2/api/images')
def m06_2_images():
    return jsonify({k: {'name': v['name']} for k, v in _fm_images.items()})

@app.route('/06-2/api/apply_all', methods=['POST'])
def m06_2_apply_all():
    data = request.json
    image_name = data.get('image_name', 'circle')
    results = nn['06-2'].apply_all_kernels(image_name, _fm_images)
    image_data = _fm_images[image_name]['data'] if image_name in _fm_images else []
    return jsonify({
        'image_data': image_data,
        'image_name': _fm_images.get(image_name, {}).get('name', ''),
        'results': results,
    })


# =============================================================
# MODULE 06-3: Pooling & Stride
# =============================================================
_cnn_images = None

@app.route('/06-3/')
def m06_3_page():
    return send_file(os.path.join(BASE, '06-cnns/02-building-a-cnn/templates/index.html'))

@app.route('/06-3/api/images')
def m06_3_images():
    global _cnn_images
    if _cnn_images is None:
        _cnn_images = nn['06-3'].make_images()
    return jsonify({k: {'name': v['name'], 'data': v['data'].tolist()} for k, v in _cnn_images.items()})

@app.route('/06-3/api/pipeline', methods=['POST'])
def m06_3_pipeline():
    data = request.json
    return jsonify(nn['06-3'].run_pipeline(data['image'], data.get('pool_mode', 'max')))

@app.route('/06-3/api/pooling', methods=['POST'])
def m06_3_pooling():
    data = request.json
    return jsonify(nn['06-3'].pooling_demo(data['image'], data.get('pool_mode', 'max')))


# =============================================================
# MODULE 07-1: Why Sequences Are Special
# =============================================================
@app.route('/07-1/')
def m07_1_page():
    return send_file(os.path.join(BASE, '07-rnns/01-why-sequences/templates/index.html'))

@app.route('/07-1/api/demo')
def m07_1_demo():
    return jsonify(nn['07-1'].order_experiment())

@app.route('/07-1/api/classify', methods=['POST'])
def m07_1_classify():
    data = request.json
    return jsonify(nn['07-1'].classify_with_mlp(data.get('sentence', 'dog bites man')))


# =============================================================
# MODULE 07-2: Recurrent Connections
# =============================================================
@app.route('/07-2/')
def m07_2_page():
    return send_file(os.path.join(BASE, '07-rnns/01-recurrence/templates/index.html'))

@app.route('/07-2/api/process', methods=['POST'])
def m07_2_process():
    data = request.json
    return jsonify(nn['07-2'].process_text(
        data.get('text', 'hello'),
        hidden_size=int(data.get('hidden_size', 16)),
    ))

@app.route('/07-2/api/gradient_flow', methods=['POST'])
def m07_2_gradient_flow():
    data = request.json
    return jsonify(nn['07-2'].gradient_flow_through_time(
        seq_length=int(data.get('seq_length', 30)),
        hidden_size=int(data.get('hidden_size', 16)),
    ))


# =============================================================
# MODULE 07-4: LSTM
# =============================================================
@app.route('/07-4/')
def m07_4_page():
    return send_file(os.path.join(BASE, '07-rnns/04-lstm/templates/index.html'))

@app.route('/07-4/api/process', methods=['POST'])
def m07_4_process():
    data = request.json
    return jsonify(nn['07-4'].process_sequence(
        data.get('text', 'hello'),
        hidden_size=int(data.get('hidden_size', 16)),
    ))

@app.route('/07-4/api/compare', methods=['POST'])
def m07_4_compare():
    data = request.json
    return jsonify(nn['07-4'].compare_gradient_flow(
        seq_length=int(data.get('seq_length', 30)),
    ))


# =============================================================
# MODULE 07-5: Build a Text Predictor
# =============================================================
_text_predictor = None

@app.route('/07-5/')
def m07_5_page():
    return send_file(os.path.join(BASE, '07-rnns/05-text-predictor/templates/index.html'))

@app.route('/07-5/api/train', methods=['POST'])
def m07_5_train():
    global _text_predictor
    data = request.json
    if _text_predictor is None:
        _text_predictor = nn['07-5'].TextPredictor()
    epochs = int(data.get('epochs', 5))
    lr = float(data.get('lr', 0.01))
    results = []
    for _ in range(epochs):
        loss = _text_predictor.train_epoch(lr)
        results.append({'epoch': _text_predictor.total_epochs, 'loss': loss})
    return jsonify({'history': results, 'total_epochs': _text_predictor.total_epochs})

@app.route('/07-5/api/generate', methods=['POST'])
def m07_5_generate():
    global _text_predictor
    if _text_predictor is None:
        _text_predictor = nn['07-5'].TextPredictor()
    data = request.json
    return jsonify(_text_predictor.generate(
        data.get('seed', 'the '),
        int(data.get('length', 100)),
        float(data.get('temperature', 0.8)),
    ))

@app.route('/07-5/api/reset', methods=['POST'])
def m07_5_reset():
    global _text_predictor
    _text_predictor = nn['07-5'].TextPredictor()
    return jsonify({'status': 'reset'})


# =============================================================
# MODULE 08-1: One-Hot Encoding
# =============================================================
@app.route('/08-1/')
def m08_1_page():
    return send_file(os.path.join(BASE, '08-embeddings/01-one-hot/templates/index.html'))

@app.route('/08-1/api/encode', methods=['POST'])
def m08_1_encode():
    data = request.json
    return jsonify(nn['08-1'].encode_sentence(data.get('sentence', 'the cat sat')))

@app.route('/08-1/api/distances')
def m08_1_distances():
    return jsonify(nn['08-1'].compare_similar_words())

@app.route('/08-1/api/sparsity')
def m08_1_sparsity():
    return jsonify(nn['08-1'].show_sparsity())

@app.route('/08-1/api/vocab')
def m08_1_vocab():
    return jsonify({'vocab': nn['08-1'].VOCAB, 'size': len(nn['08-1'].VOCAB)})


# =============================================================
# BOOT
# =============================================================
if __name__ == '__main__':
    ready = [m for m in MODULES if m['status'] == 'ready']
    print()
    print("  +=============================================+")
    print("  |  Neural Networks: From Zero to Transformers |")
    print("  |  Open http://localhost:5000 in browser      |")
    print("  +=============================================+")
    print(f"  |  {len(ready)} lessons ready                          |")
    print("  +=============================================+")
    print()
    app.run(debug=True, port=5000)
