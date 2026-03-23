"""Module 08.4: How Embeddings Actually Work"""
from flask import Flask, render_template, jsonify, request
from nn_code import demo_forward, demo_training, real_world_comparison, VOCAB

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forward', methods=['POST'])
def forward():
    data = request.json
    return jsonify(demo_forward(data.get('word', 'cat')))

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    return jsonify(demo_training(
        num_epochs=int(data.get('epochs', 30)),
        lr=float(data.get('lr', 0.15)),
    ))

@app.route('/api/real_world')
def real_world():
    return jsonify(real_world_comparison())

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 08.4: How Embeddings Work        |")
    print("  |  Open http://localhost:5033 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5033)
