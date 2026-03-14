"""Module 05.2: Optimizers — SGD, Momentum, RMSProp, Adam"""
from flask import Flask, render_template, jsonify, request
from nn_code import run_optimizers, nn_comparison, LANDSCAPES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/landscape', methods=['POST'])
def landscape():
    data = request.json
    return jsonify(run_optimizers(
        landscape=data.get('landscape', 'bowl'),
        lr=float(data.get('lr', 0.01)),
        steps=int(data.get('steps', 150)),
    ))

@app.route('/api/nn_compare', methods=['POST'])
def nn_compare():
    data = request.json
    return jsonify(nn_comparison(
        epochs=int(data.get('epochs', 50)),
        lr=float(data.get('lr', 0.5)),
    ))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 05.2: Optimizers                 |")
    print("  |  Open http://localhost:5013 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5013)
