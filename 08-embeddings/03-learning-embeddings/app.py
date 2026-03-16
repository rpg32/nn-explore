"""Module 08.3: Learning Embeddings — Word2Vec from scratch"""
from flask import Flask, render_template, jsonify, request
from nn_code import train_embeddings, CORPUS, VOCAB

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    return jsonify(train_embeddings(
        epochs=int(data.get('epochs', 100)),
        embed_dim=2,
        lr=float(data.get('lr', 0.1)),
    ))

@app.route('/api/info')
def info():
    return jsonify({'corpus': CORPUS[:500], 'vocab': VOCAB, 'vocab_size': len(VOCAB)})

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 08.3: Learning Embeddings        |")
    print("  |  Open http://localhost:5029 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5029)
