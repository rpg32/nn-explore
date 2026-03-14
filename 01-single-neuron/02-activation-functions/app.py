"""
Module 01.2: Activation Functions - Interactive Demo

Run this file:
    python app.py

Then open http://localhost:5002 in your browser.
"""

from flask import Flask, render_template, jsonify, request
from nn_code import compute_activation, compute_curve, demonstrate_linearity_problem, ACTIVATIONS

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/activate', methods=['POST'])
def activate():
    """Compute activation for a single value across all functions."""
    data = request.json
    x = float(data['x'])
    results = {}
    for name in ACTIVATIONS:
        results[name] = compute_activation(name, x)
    return jsonify(results)


@app.route('/api/curves')
def curves():
    """Return the full curve data for all activation functions."""
    result = {}
    for name in ACTIVATIONS:
        result[name] = compute_curve(name)
    return jsonify(result)


@app.route('/api/linearity')
def linearity():
    """Return data demonstrating the linearity problem."""
    return jsonify(demonstrate_linearity_problem())


if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 01.2: Activation Functions       |")
    print("  |  Open http://localhost:5002 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5002)
