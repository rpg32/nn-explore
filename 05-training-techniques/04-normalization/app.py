"""Module 05.4: Data Normalization"""
from flask import Flask, render_template, jsonify, request
from nn_code import train_comparison

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.json
    return jsonify(train_comparison(
        epochs=int(data.get('epochs', 100)),
        lr=float(data.get('lr', 0.5)),
        scale_ratio=float(data.get('scale_ratio', 100)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 05.4: Data Normalization         |")
    print("  |  Open http://localhost:5015 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5015)
