"""Module 06.2: Feature Maps & Filters"""
from flask import Flask, render_template, jsonify, request
from nn_code import make_test_images, apply_all_kernels, KERNELS

app = Flask(__name__)

# Pre-generate images once at startup
IMAGES = make_test_images()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/images', methods=['GET'])
def images():
    """Return available test images (names only, no pixel data)."""
    return jsonify({k: {'name': v['name']} for k, v in IMAGES.items()})


@app.route('/api/apply_all', methods=['POST'])
def apply_all():
    """Apply ALL 8 kernels to the chosen image, return all feature maps."""
    data = request.json
    image_name = data.get('image_name', 'circle')
    results = apply_all_kernels(image_name, IMAGES)

    # Also send the raw image data for display
    image_data = IMAGES[image_name]['data'] if image_name in IMAGES else []
    return jsonify({
        'image_data': image_data,
        'image_name': IMAGES.get(image_name, {}).get('name', ''),
        'results': results,
    })


if __name__ == '__main__':
    print("  +==========================================+")
    print("  |  Module 06.2: Feature Maps & Filters     |")
    print("  |  Open http://localhost:5023 in browser    |")
    print("  +==========================================+")
    app.run(debug=True, port=5023)
