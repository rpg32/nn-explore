"""Module 03.3: Gradient Flow"""
from flask import Flask, render_template, jsonify, request
from nn_code import simulate_gradient_flow, compare_activations, depth_sweep

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    return jsonify(compare_activations(
        int(data['num_layers']), float(data.get('weight_scale', 1.0))))

@app.route('/api/sweep', methods=['POST'])
def sweep():
    data = request.json
    return jsonify({
        'sigmoid': depth_sweep('sigmoid', float(data.get('weight_scale', 1.0))),
        'relu': depth_sweep('relu', float(data.get('weight_scale', 1.0))),
    })

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 03.3: Gradient Flow              |")
    print("  |  Open http://localhost:5010 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5010)
