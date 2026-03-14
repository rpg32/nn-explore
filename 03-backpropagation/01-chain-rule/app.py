"""Module 03.1: The Chain Rule - Interactive Demo"""
from flask import Flask, render_template, jsonify, request
from nn_code import compute_graph

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compute', methods=['POST'])
def compute():
    data = request.json
    return jsonify(compute_graph(
        float(data['x']), float(data['w']),
        float(data['b']), float(data['y_true'])))

if __name__ == '__main__':
    print()
    print("  +==========================================+")
    print("  |  Module 03.1: The Chain Rule             |")
    print("  |  Open http://localhost:5008 in browser   |")
    print("  +==========================================+")
    print()
    app.run(debug=True, port=5008)
