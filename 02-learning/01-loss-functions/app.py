"""
Module 02.1: Loss Functions - Interactive Demo
"""

from flask import Flask, render_template, jsonify, request
from nn_code import compute_both, mse_curve, bce_curve, compute_batch_loss, SAMPLE_BATCH

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/compute', methods=['POST'])
def compute():
    data = request.json
    y_true = float(data['y_true'])
    y_pred = float(data['y_pred'])
    result = compute_both(y_true, y_pred)
    result['mse_curve'] = mse_curve(y_true)
    result['bce_curve'] = bce_curve(y_true)
    return jsonify(result)


@app.route('/api/batch', methods=['POST'])
def batch():
    data = request.json
    return jsonify(compute_batch_loss(data['predictions'], data.get('loss_type', 'bce')))


@app.route('/api/sample_batch')
def sample_batch():
    return jsonify(SAMPLE_BATCH)


if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 02.1: Loss Functions             |")
    print("  |  Open http://localhost:5004 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5004)
