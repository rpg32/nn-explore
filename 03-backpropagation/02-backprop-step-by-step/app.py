"""Module 03.2: Backprop Step-by-Step"""
from flask import Flask, render_template, jsonify, request
from nn_code import TinyNetwork

app = Flask(__name__)
net = TinyNetwork()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forward', methods=['POST'])
def forward():
    data = request.json
    return jsonify(net.forward_detailed(data['x'], float(data['y_true'])))

@app.route('/api/backward', methods=['POST'])
def backward():
    data = request.json
    return jsonify(net.backward_detailed(data['x'], float(data['y_true'])))

@app.route('/api/train_step', methods=['POST'])
def train_step():
    data = request.json
    return jsonify(net.train_step(data['x'], float(data['y_true']), float(data['lr'])))

@app.route('/api/train_many', methods=['POST'])
def train_many():
    data = request.json
    return jsonify(net.train_many(data['x'], float(data['y_true']),
                                  float(data['lr']), int(data.get('steps', 50))))

@app.route('/api/reset', methods=['POST'])
def reset():
    net.reset()
    return jsonify({'params': net.get_params()})

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 03.2: Backprop Step-by-Step      |")
    print("  |  Open http://localhost:5009 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5009)
