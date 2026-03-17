"""Module 09.3: Attention Scores & Weights"""
from flask import Flask, render_template, jsonify, request
from nn_code import compute_attention_matrix, EXAMPLE_SENTENCES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compute', methods=['POST'])
def compute():
    data = request.json
    return jsonify(compute_attention_matrix(data.get('sentence', 'the cat sat on the mat')))

@app.route('/api/examples')
def examples():
    return jsonify({'sentences': EXAMPLE_SENTENCES})

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 09.3: Scores & Weights           |")
    print("  |  Open http://localhost:5032 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5032)
