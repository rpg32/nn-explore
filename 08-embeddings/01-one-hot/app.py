"""Module 08.1: One-Hot Encoding"""
from flask import Flask, render_template, jsonify, request
from nn_code import (one_hot_encode, encode_sentence, compute_distances,
                     compare_similar_words, show_sparsity, VOCAB)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/encode', methods=['POST'])
def encode():
    data = request.json
    return jsonify(encode_sentence(data.get('sentence', 'the cat sat')))

@app.route('/api/distances')
def distances():
    return jsonify(compare_similar_words())

@app.route('/api/sparsity')
def sparsity():
    return jsonify(show_sparsity())

@app.route('/api/vocab')
def vocab():
    return jsonify({'vocab': VOCAB, 'size': len(VOCAB)})

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 08.1: One-Hot Encoding           |")
    print("  |  Open http://localhost:5027 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5027)
