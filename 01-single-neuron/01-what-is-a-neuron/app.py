"""
Module 01.1: What is a Neuron? — Interactive Demo

Run this file:
    python app.py

Then open http://localhost:5001 in your browser.
"""

from flask import Flask, render_template, jsonify, request
from nn_code import Neuron

app = Flask(__name__)

# Create our neuron with 3 inputs
neuron = Neuron(num_inputs=3)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/compute', methods=['POST'])
def compute():
    """Receive inputs/weights/bias from the browser, run through our Python neuron."""
    data = request.json

    neuron.set_weights(data['weights'])
    neuron.set_bias(data['bias'])

    result = neuron.forward(data['inputs'])
    return jsonify(result)


if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 01.1: What is a Neuron?          |")
    print("  |  Open http://localhost:5001 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5001)
