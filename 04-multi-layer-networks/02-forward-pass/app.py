"""Module 04.2: The Forward Pass — Step by step through the network"""
from flask import Flask, render_template, jsonify, request
from nn_code import run_forward_pass

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forward', methods=['POST'])
def forward():
    data = request.json
    return jsonify(run_forward_pass(
        data['architecture'],
        data.get('inputs', [0.5, -0.3]),
        seed=int(data.get('seed', 42)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 04.2: The Forward Pass           |")
    print("  |  Open http://localhost:5021 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5021)
