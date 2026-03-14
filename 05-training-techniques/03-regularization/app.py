"""Module 05.3: Overfitting & Regularization"""
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
        epochs=int(data.get('epochs', 200)),
        lr=float(data.get('lr', 0.5)),
        l2_lambda=float(data.get('l2_lambda', 0.01)),
        dropout_rate=float(data.get('dropout_rate', 0.5)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 05.3: Regularization             |")
    print("  |  Open http://localhost:5014 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5014)
