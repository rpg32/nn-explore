"""Module 04.1: Layers — Input, Hidden, Output"""
from flask import Flask, render_template, jsonify, request
from nn_code import demo_forward

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forward', methods=['POST'])
def forward():
    data = request.json
    return jsonify(demo_forward(
        data['architecture'],
        data.get('inputs', [0.5, -0.3]),
        seed=int(data.get('seed', 42)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 04.1: Layers                     |")
    print("  |  Open http://localhost:5020 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5020)
