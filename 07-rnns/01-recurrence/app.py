"""Module 07: RNNs — Recurrence and Sequential Processing"""
from flask import Flask, render_template, jsonify, request
from nn_code import process_text, gradient_flow_through_time, EXAMPLE_TEXTS

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    data = request.json
    return jsonify(process_text(
        data.get('text', 'hello'),
        hidden_size=int(data.get('hidden_size', 16)),
    ))

@app.route('/api/gradient_flow', methods=['POST'])
def grad_flow():
    data = request.json
    return jsonify(gradient_flow_through_time(
        seq_length=int(data.get('seq_length', 30)),
        hidden_size=int(data.get('hidden_size', 16)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 07: RNNs                         |")
    print("  |  Open http://localhost:5018 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5018)
