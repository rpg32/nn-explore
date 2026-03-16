"""Module 07.4: LSTM — Long Short-Term Memory"""
from flask import Flask, render_template, jsonify, request
from nn_code import process_sequence, compare_gradient_flow

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    """Process text through the LSTM, returning all gate values at each step."""
    data = request.json
    return jsonify(process_sequence(
        data.get('text', 'hello'),
        hidden_size=int(data.get('hidden_size', 16)),
    ))

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare gradient flow in RNN vs LSTM."""
    data = request.json
    return jsonify(compare_gradient_flow(
        seq_length=int(data.get('seq_length', 40)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 07.4: LSTM                       |")
    print("  |  Open http://localhost:5025 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5025)
