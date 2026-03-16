"""Module 08.2: Dense Embeddings"""
from flask import Flask, render_template, jsonify, request
from nn_code import (get_all_embeddings, compute_nearest, analogy,
                     compare_distances, DEFAULT_PAIRS)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/embeddings')
def embeddings():
    return jsonify(get_all_embeddings())

@app.route('/api/nearest', methods=['POST'])
def nearest():
    data = request.json
    return jsonify(compute_nearest(data.get('word', 'cat'), int(data.get('n', 5))))

@app.route('/api/analogy', methods=['POST'])
def analogy_route():
    data = request.json
    return jsonify(analogy(data['a'], data['b'], data['c']))

@app.route('/api/distances')
def distances():
    return jsonify(compare_distances(DEFAULT_PAIRS))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 08.2: Dense Embeddings           |")
    print("  |  Open http://localhost:5028 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5028)
