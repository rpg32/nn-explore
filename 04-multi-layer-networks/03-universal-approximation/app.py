"""Module 04.3: Universal Approximation — Fit any function with enough neurons"""
from flask import Flask, render_template, jsonify, request
from nn_code import fit_with_increasing_neurons

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fit', methods=['POST'])
def fit():
    data = request.json
    function_name = data.get('function_name', 'sine')
    max_neurons = int(data.get('max_neurons', 64))
    epochs = int(data.get('epochs', 3000))

    # Build neuron counts up to max
    all_counts = [1, 2, 4, 8, 16, 32, 64]
    neuron_counts = [n for n in all_counts if n <= max_neurons]

    result = fit_with_increasing_neurons(
        function_name,
        neuron_counts=neuron_counts,
        epochs=epochs,
    )
    return jsonify(result)

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 04.3: Universal Approximation    |")
    print("  |  Open http://localhost:5022 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5022)
