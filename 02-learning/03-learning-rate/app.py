"""Module 02.3: Learning Rate - Interactive Demo"""

from flask import Flask, render_template, jsonify, request
from nn_code import run_three, compute_landscape, lr_sweep

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/landscape')
def landscape():
    return jsonify(compute_landscape())

@app.route('/api/run', methods=['POST'])
def run():
    data = request.json
    return jsonify(run_three(
        data['w1'], data['w2'],
        data['lr_slow'], data['lr_good'], data['lr_fast'],
        data.get('steps', 60)))

@app.route('/api/sweep', methods=['POST'])
def sweep():
    data = request.json
    return jsonify(lr_sweep(data['w1'], data['w2']))

if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 02.3: Learning Rate              |")
    print("  |  Open http://localhost:5006 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5006)
