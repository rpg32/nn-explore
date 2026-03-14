"""Module 06.2: Building a CNN — Pooling + Full Pipeline"""
from flask import Flask, render_template, jsonify, request
from nn_code import make_images, run_pipeline, pooling_demo

app = Flask(__name__)
_images = make_images()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def images():
    return jsonify({k: {'name': v['name'], 'data': v['data'].tolist()} for k, v in _images.items()})

@app.route('/api/pipeline', methods=['POST'])
def pipeline():
    data = request.json
    return jsonify(run_pipeline(data['image'], data.get('pool_mode', 'max')))

@app.route('/api/pooling', methods=['POST'])
def pooling():
    data = request.json
    return jsonify(pooling_demo(data['image'], data.get('pool_mode', 'max')))

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 06.2: Building a CNN             |")
    print("  |  Open http://localhost:5017 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5017)
