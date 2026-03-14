"""Module 05.1: Batch vs Stochastic Gradient Descent"""
from flask import Flask, render_template, jsonify, request
from nn_code import train_comparison, train_step_by_step

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.json
    return jsonify(train_comparison(
        num_epochs=int(data.get('epochs', 30)),
        lr=float(data.get('lr', 0.5)),
        batch_size=int(data.get('batch_size', 16)),
    ))

@app.route('/api/traces', methods=['POST'])
def traces():
    data = request.json
    return jsonify(train_step_by_step(
        total_epochs=int(data.get('epochs', 30)),
        lr=float(data.get('lr', 0.5)),
        batch_size=int(data.get('batch_size', 16)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 05.1: Batch vs SGD               |")
    print("  |  Open http://localhost:5012 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5012)
