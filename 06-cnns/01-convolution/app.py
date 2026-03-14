"""Module 06.1: Convolution"""
from flask import Flask, render_template, jsonify, request
from nn_code import make_test_images, apply_kernel, compute_single_step, KERNELS

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def images():
    return jsonify(make_test_images())

@app.route('/api/kernels')
def kernels():
    return jsonify({k: {'name': v['name'], 'desc': v['desc'], 'kernel': v['kernel']}
                    for k, v in KERNELS.items()})

@app.route('/api/apply', methods=['POST'])
def apply():
    data = request.json
    return jsonify(apply_kernel(data['image'], data['kernel']))

@app.route('/api/step', methods=['POST'])
def step():
    data = request.json
    result = compute_single_step(data['image'], data['kernel'], data['row'], data['col'])
    return jsonify(result) if result else ('', 400)

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 06.1: Convolution                |")
    print("  |  Open http://localhost:5016 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5016)
