"""
Module 02.2: Gradient Descent - Interactive Demo
"""

from flask import Flask, render_template, jsonify, request
from nn_code import compute_landscape, gradient_descent_step, run_descent, LANDSCAPES

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/landscape/<name>')
def landscape(name):
    if name not in LANDSCAPES:
        return jsonify({'error': 'Unknown landscape'}), 404
    return jsonify(compute_landscape(name))


@app.route('/api/step', methods=['POST'])
def step():
    data = request.json
    return jsonify(gradient_descent_step(
        data['landscape'], data['w1'], data['w2'], data['lr']))


@app.route('/api/run', methods=['POST'])
def run():
    data = request.json
    return jsonify(run_descent(
        data['landscape'], data['w1'], data['w2'],
        data['lr'], data.get('steps', 50)))


@app.route('/api/landscapes')
def list_landscapes():
    return jsonify({k: {'label': v['label'], 'desc': v['desc']}
                    for k, v in LANDSCAPES.items()})


if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 02.2: Gradient Descent           |")
    print("  |  Open http://localhost:5005 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5005)
