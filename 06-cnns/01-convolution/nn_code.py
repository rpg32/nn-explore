"""
=== MODULE 06.1: CONVOLUTION ===

So far, our neurons look at ALL inputs at once.
For a 28x28 image (784 pixels), every neuron connects
to all 784 pixels. This has problems:

  1. Too many parameters (784 × neurons in first layer)
  2. No spatial awareness — a pixel at (0,0) and (27,27)
     are treated identically
  3. No translation invariance — learning "cat at top-left"
     doesn't help recognize "cat at bottom-right"

CONVOLUTION fixes this. Instead of connecting to everything,
a KERNEL (small filter, e.g. 3×3) slides across the image:

  At each position:
    output[i,j] = sum(kernel * image_patch[i:i+3, j:j+3])

This is EXACTLY a weighted sum — just like a neuron!
But only over a small local region. And the SAME weights
(kernel) are shared across all positions.

  image_patch:     kernel:          output:
  [10, 20, 30]     [1, 0,-1]
  [40, 50, 60]  ×  [1, 0,-1]  →   sum = single number
  [70, 80, 90]     [1, 0,-1]

The kernel slides across every position → output is a
2D "feature map" showing WHERE the pattern was detected.

DIFFERENT KERNELS DETECT DIFFERENT FEATURES:
  Horizontal edge kernel: detects horizontal edges
  Vertical edge kernel:   detects vertical edges
  Blur kernel:            smooths the image
  Sharpen kernel:         enhances detail

In a CNN, the network LEARNS what kernels to use.
It starts random and backprop adjusts the kernel weights
to detect whatever features help classification.
"""

import numpy as np


# ============================================================
# CONVOLUTION OPERATION (from scratch)
# ============================================================
def convolve2d(image, kernel):
    """2D convolution — the fundamental CNN operation.

    image: 2D array (H, W)
    kernel: 2D array (kH, kW), typically 3×3 or 5×5

    Returns: feature map of shape (H-kH+1, W-kW+1)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel)

    return output


# ============================================================
# PREDEFINED KERNELS
# ============================================================
KERNELS = {
    'edge_h': {
        'name': 'Horizontal Edge',
        'desc': 'Detects horizontal boundaries — bright above, dark below',
        'kernel': [[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]],
    },
    'edge_v': {
        'name': 'Vertical Edge',
        'desc': 'Detects vertical boundaries — bright left, dark right',
        'kernel': [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]],
    },
    'edge_all': {
        'name': 'All Edges (Laplacian)',
        'desc': 'Detects edges in all directions — highlights any sharp change',
        'kernel': [[ 0, -1,  0],
                   [-1,  4, -1],
                   [ 0, -1,  0]],
    },
    'sharpen': {
        'name': 'Sharpen',
        'desc': 'Enhances detail by amplifying differences from neighbors',
        'kernel': [[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]],
    },
    'blur': {
        'name': 'Blur (Average)',
        'desc': 'Smooths the image by averaging each pixel with its neighbors',
        'kernel': [[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]],
    },
    'emboss': {
        'name': 'Emboss',
        'desc': 'Creates a 3D shadow effect — highlights directional gradients',
        'kernel': [[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]],
    },
    'identity': {
        'name': 'Identity',
        'desc': 'Does nothing — output equals input. Baseline for comparison.',
        'kernel': [[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]],
    },
}


# ============================================================
# TEST IMAGES (generated, no file dependencies)
# ============================================================
def make_test_images():
    """Generate simple test images that show convolution effects clearly."""
    images = {}
    sz = 28  # Same size as MNIST

    # Horizontal stripes
    img = np.zeros((sz, sz))
    for i in range(sz):
        if (i // 4) % 2 == 0:
            img[i, :] = 1.0
    images['h_stripes'] = {'name': 'Horizontal Stripes', 'data': img.tolist()}

    # Vertical stripes
    img = np.zeros((sz, sz))
    for j in range(sz):
        if (j // 4) % 2 == 0:
            img[:, j] = 1.0
    images['v_stripes'] = {'name': 'Vertical Stripes', 'data': img.tolist()}

    # Checkerboard
    img = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(sz):
            if ((i // 4) + (j // 4)) % 2 == 0:
                img[i, j] = 1.0
    images['checker'] = {'name': 'Checkerboard', 'data': img.tolist()}

    # Circle
    img = np.zeros((sz, sz))
    cx, cy = sz // 2, sz // 2
    for i in range(sz):
        for j in range(sz):
            if (i - cy)**2 + (j - cx)**2 < (sz * 0.35)**2:
                img[i, j] = 1.0
    images['circle'] = {'name': 'Circle', 'data': img.tolist()}

    # Diagonal line
    img = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(max(0, i-1), min(sz, i+2)):
            img[i, j] = 1.0
    images['diagonal'] = {'name': 'Diagonal', 'data': img.tolist()}

    # Cross / plus sign
    img = np.zeros((sz, sz))
    mid = sz // 2
    img[mid-2:mid+2, 4:sz-4] = 1.0
    img[4:sz-4, mid-2:mid+2] = 1.0
    images['cross'] = {'name': 'Cross', 'data': img.tolist()}

    # Digit-like "7"
    img = np.zeros((sz, sz))
    img[4:7, 6:22] = 1.0  # top bar
    for i in range(7, 24):
        col = int(22 - (i - 7) * 0.8)
        img[i, max(0, col-1):min(sz, col+2)] = 1.0
    images['seven'] = {'name': 'Digit "7"', 'data': img.tolist()}

    return images


def apply_kernel(image_data, kernel_data):
    """Apply a kernel to an image, return the feature map and step-by-step data."""
    image = np.array(image_data)
    kernel = np.array(kernel_data)
    feature_map = convolve2d(image, kernel)

    # Normalize for display (map to 0-1)
    fm_display = feature_map.copy()
    if fm_display.max() != fm_display.min():
        fm_display = (fm_display - fm_display.min()) / (fm_display.max() - fm_display.min())
    else:
        fm_display = np.zeros_like(fm_display)

    return {
        'feature_map': feature_map.tolist(),
        'feature_map_display': fm_display.tolist(),
        'fm_min': float(feature_map.min()),
        'fm_max': float(feature_map.max()),
        'fm_shape': list(feature_map.shape),
    }


def compute_single_step(image_data, kernel_data, row, col):
    """Show the computation for a single position — the weighted sum."""
    image = np.array(image_data)
    kernel = np.array(kernel_data)
    kH, kW = kernel.shape

    if row + kH > image.shape[0] or col + kW > image.shape[1]:
        return None

    patch = image[row:row+kH, col:col+kW]
    products = (patch * kernel).tolist()
    result = float(np.sum(patch * kernel))

    return {
        'patch': patch.tolist(),
        'kernel': kernel.tolist(),
        'products': products,
        'sum': result,
        'row': row, 'col': col,
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  CONVOLUTION")
    print("=" * 55)

    images = make_test_images()
    for kname, kinfo in KERNELS.items():
        kernel = np.array(kinfo['kernel'])
        img = np.array(images['circle']['data'])
        fm = convolve2d(img, kernel)
        print(f"  {kname:10s}: input {img.shape} → output {fm.shape}  "
              f"range [{fm.min():.2f}, {fm.max():.2f}]")
