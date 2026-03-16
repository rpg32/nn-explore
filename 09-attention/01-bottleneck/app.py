"""Module 09.1: The Bottleneck Problem"""
from flask import Flask, render_template, jsonify, request
from nn_code import information_retention, SENTENCES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    return jsonify(information_retention(
        hidden_size=int(data.get('hidden_size', 16)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 09.1: The Bottleneck Problem     |")
    print("  |  Open http://localhost:5030 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5030)
