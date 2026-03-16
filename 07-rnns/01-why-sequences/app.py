"""Module 07.1: Why Sequences Are Special"""
from flask import Flask, render_template, jsonify, request
from nn_code import order_experiment, time_series_demo, compare_sentences

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/demo')
def demo():
    """Return the full order experiment results."""
    return jsonify(order_experiment())

@app.route('/api/timeseries')
def timeseries():
    """Return time series pattern comparisons."""
    return jsonify(time_series_demo())

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare two user-provided sentences through the MLP."""
    data = request.json
    sentence_a = data.get('sentence_a', 'dog bites man')
    sentence_b = data.get('sentence_b', 'man bites dog')
    return jsonify(compare_sentences(sentence_a, sentence_b))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 07.1: Why Sequences Are Special  |")
    print("  |  Open http://localhost:5024 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5024)
