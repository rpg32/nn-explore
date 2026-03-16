"""Module 09.2: Attention as Weighted Lookup"""
from flask import Flask, render_template, jsonify, request
from nn_code import run_example, EXAMPLES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/example', methods=['POST'])
def example():
    data = request.json
    return jsonify(run_example(data.get('example', 'animal_lookup')))

@app.route('/api/examples')
def examples():
    return jsonify({k: {'name': v['name'], 'desc': v['desc']} for k, v in EXAMPLES.items()})

if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 09.2: Attention Weighted Lookup  |")
    print("  |  Open http://localhost:5031 in browser   |")
    print("  +==========================================+")
    app.run(debug=True, port=5031)
