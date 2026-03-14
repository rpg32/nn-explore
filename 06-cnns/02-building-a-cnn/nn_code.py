"""
=== MODULE 06.2: BUILDING A CNN ===

Now we assemble the full pipeline:

  Image → Conv+ReLU → Pool → Conv+ReLU → Pool → Flatten → MLP → Class

POOLING — why and how:

  After convolution, the feature maps are almost the same size as the
  input (28×28 → 26×26). We need to shrink them because:
    1. Reduce computation for deeper layers
    2. Build TRANSLATION INVARIANCE — if a feature moves a pixel
       or two, the pooled output doesn't change
    3. Increase receptive field — deeper kernels effectively "see"
       a larger region of the original image

  MAX POOLING (most common):
    Take a 2×2 window, keep ONLY the maximum value.
    26×26 → 13×13 (half the size)
    If the feature was detected anywhere in the 2×2 region, it survives.

  AVERAGE POOLING:
    Take a 2×2 window, average the values.
    Smoother but less sharp than max pooling.

THE FULL CNN PIPELINE:

  Input:     28×28×1   (grayscale image)
  Conv1:     4 kernels (3×3) → 26×26×4   (4 edge-type feature maps)
  ReLU:      zero out negatives
  MaxPool:   2×2 → 13×13×4
  Conv2:     8 kernels (3×3) → 11×11×8   (8 higher-level features)
  ReLU:      zero out negatives
  MaxPool:   2×2 → 5×5×8
  Flatten:   5×5×8 = 200 numbers
  Dense:     200 → 10 (one per digit class)
  Softmax:   probabilities that sum to 1

  The convolution kernels are LEARNED via backpropagation,
  just like the weights in our MLP. We just use fixed kernels
  here so you can see what each layer does.
"""

import numpy as np


# ============================================================
# OPERATIONS
# ============================================================
def convolve2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    out = np.zeros((H - kH + 1, W - kW + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return out


def relu(x):
    return np.maximum(0, x)


def max_pool2d(image, size=2):
    """Max pooling with given window size."""
    H, W = image.shape
    out_h = H // size
    out_w = W // size
    out = np.zeros((out_h, out_w))
    which = np.zeros((out_h, out_w, 2), dtype=int)  # track which position won
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*size:(i+1)*size, j*size:(j+1)*size]
            out[i, j] = patch.max()
            idx = np.unravel_index(patch.argmax(), patch.shape)
            which[i, j] = [i*size + idx[0], j*size + idx[1]]
    return out, which


def avg_pool2d(image, size=2):
    H, W = image.shape
    out_h = H // size
    out_w = W // size
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = image[i*size:(i+1)*size, j*size:(j+1)*size].mean()
    return out


def normalize_map(fm):
    """Normalize a feature map to [0, 1] for display."""
    mn, mx = fm.min(), fm.max()
    if mx - mn < 1e-8:
        return np.zeros_like(fm)
    return (fm - mn) / (mx - mn)


# ============================================================
# FIXED KERNELS (simulating what a trained CNN might learn)
# ============================================================
LAYER1_KERNELS = [
    {'name': 'H-Edge',  'kernel': np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=float)},
    {'name': 'V-Edge',  'kernel': np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=float)},
    {'name': 'Diag \\', 'kernel': np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float) / 3},
    {'name': 'Diag /',  'kernel': np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float) / 3},
]

# Layer 2 kernels operate on the 4 feature maps from layer 1
# Each kernel is 3×3×4 (3×3 spatial, across 4 input channels)
# For simplicity, we define them as combinations
LAYER2_KERNELS = [
    {'name': 'Corner TL', 'weights': [
        np.array([[0,0,0],[0,1,1],[0,1,0]], dtype=float),   # H-edge
        np.array([[0,0,0],[0,1,0],[0,1,1]], dtype=float),   # V-edge
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
    ]},
    {'name': 'Corner TR', 'weights': [
        np.array([[0,0,0],[1,1,0],[0,1,0]], dtype=float),
        np.array([[0,0,0],[0,1,0],[1,1,0]], dtype=float),
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
    ]},
    {'name': 'Curve', 'weights': [
        np.array([[0,1,0],[0,1,0],[0,0,0]], dtype=float),
        np.array([[0,0,0],[1,1,0],[0,0,0]], dtype=float),
        np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float),
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
    ]},
    {'name': 'Cross', 'weights': [
        np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=float),
        np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=float),
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
        np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=float),
    ]},
]


# ============================================================
# TEST IMAGES
# ============================================================
def make_images():
    images = {}
    sz = 28

    # Circle
    img = np.zeros((sz, sz))
    cx, cy = sz // 2, sz // 2
    for i in range(sz):
        for j in range(sz):
            if (i - cy)**2 + (j - cx)**2 < (sz * 0.35)**2:
                img[i, j] = 1.0
    images['circle'] = {'name': 'Circle', 'data': img}

    # Cross
    img = np.zeros((sz, sz))
    mid = sz // 2
    img[mid-2:mid+2, 4:sz-4] = 1.0
    img[4:sz-4, mid-2:mid+2] = 1.0
    images['cross'] = {'name': 'Cross', 'data': img}

    # Digit "7"
    img = np.zeros((sz, sz))
    img[4:7, 6:22] = 1.0
    for i in range(7, 24):
        col = int(22 - (i - 7) * 0.8)
        img[i, max(0, col-1):min(sz, col+2)] = 1.0
    images['seven'] = {'name': 'Digit "7"', 'data': img}

    # Digit "0"
    img = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(sz):
            r = ((i - 14)**2 / 80 + (j - 14)**2 / 40)
            if 0.5 < r < 1.2:
                img[i, j] = 1.0
    images['zero'] = {'name': 'Digit "0"', 'data': img}

    # L-shape
    img = np.zeros((sz, sz))
    img[4:22, 6:10] = 1.0
    img[18:22, 6:20] = 1.0
    images['L'] = {'name': 'L-Shape', 'data': img}

    return images


# ============================================================
# FULL CNN PIPELINE
# ============================================================
def run_pipeline(image_data, pool_mode='max'):
    """Run the full CNN pipeline and return all intermediate results."""
    image = np.array(image_data)

    results = {
        'input': {'data': normalize_map(image).tolist(), 'shape': list(image.shape)},
    }

    # === LAYER 1: Conv + ReLU ===
    conv1_maps = []
    for k in LAYER1_KERNELS:
        fm = convolve2d(image, k['kernel'])
        fm = relu(fm)
        conv1_maps.append(fm)

    results['conv1'] = {
        'maps': [normalize_map(fm).tolist() for fm in conv1_maps],
        'names': [k['name'] for k in LAYER1_KERNELS],
        'shape': list(conv1_maps[0].shape),
        'num': len(conv1_maps),
    }

    # === POOL 1: 2×2 ===
    pool1_maps = []
    for fm in conv1_maps:
        if pool_mode == 'max':
            pooled, _ = max_pool2d(fm)
        else:
            pooled = avg_pool2d(fm)
        pool1_maps.append(pooled)

    results['pool1'] = {
        'maps': [normalize_map(fm).tolist() for fm in pool1_maps],
        'shape': list(pool1_maps[0].shape),
    }

    # === LAYER 2: Conv + ReLU (multi-channel) ===
    conv2_maps = []
    for k2 in LAYER2_KERNELS:
        combined = np.zeros((pool1_maps[0].shape[0] - 2, pool1_maps[0].shape[1] - 2))
        for ch, w in enumerate(k2['weights']):
            combined += convolve2d(pool1_maps[ch], w)
        combined = relu(combined)
        conv2_maps.append(combined)

    results['conv2'] = {
        'maps': [normalize_map(fm).tolist() for fm in conv2_maps],
        'names': [k['name'] for k in LAYER2_KERNELS],
        'shape': list(conv2_maps[0].shape) if conv2_maps else [0, 0],
        'num': len(conv2_maps),
    }

    # === POOL 2: 2×2 ===
    pool2_maps = []
    for fm in conv2_maps:
        if fm.shape[0] >= 2 and fm.shape[1] >= 2:
            if pool_mode == 'max':
                pooled, _ = max_pool2d(fm)
            else:
                pooled = avg_pool2d(fm)
            pool2_maps.append(pooled)

    results['pool2'] = {
        'maps': [normalize_map(fm).tolist() for fm in pool2_maps],
        'shape': list(pool2_maps[0].shape) if pool2_maps else [0, 0],
    }

    # === FLATTEN ===
    flat = np.concatenate([fm.ravel() for fm in pool2_maps])
    results['flatten'] = {
        'length': len(flat),
        'values': flat.tolist()[:50],  # first 50 for display
    }

    # Summary
    results['summary'] = {
        'stages': [
            {'name': 'Input', 'shape': f'{image.shape[0]}×{image.shape[1]}×1',
             'params': 0},
            {'name': 'Conv1+ReLU', 'shape': f'{conv1_maps[0].shape[0]}×{conv1_maps[0].shape[1]}×{len(conv1_maps)}',
             'params': sum(k['kernel'].size for k in LAYER1_KERNELS)},
            {'name': 'MaxPool', 'shape': f'{pool1_maps[0].shape[0]}×{pool1_maps[0].shape[1]}×{len(pool1_maps)}',
             'params': 0},
            {'name': 'Conv2+ReLU', 'shape': f'{conv2_maps[0].shape[0]}×{conv2_maps[0].shape[1]}×{len(conv2_maps)}' if conv2_maps else '?',
             'params': sum(sum(w.size for w in k['weights']) for k in LAYER2_KERNELS)},
            {'name': 'MaxPool', 'shape': f'{pool2_maps[0].shape[0]}×{pool2_maps[0].shape[1]}×{len(pool2_maps)}' if pool2_maps else '?',
             'params': 0},
            {'name': 'Flatten', 'shape': str(len(flat)),
             'params': 0},
            {'name': 'Dense->10', 'shape': '10',
             'params': len(flat) * 10 + 10},
        ],
    }

    return results


def pooling_demo(image_data, pool_mode='max'):
    """Focused demo of just the pooling operation."""
    image = np.array(image_data)
    # First apply an edge kernel to have interesting values
    kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=float)
    fm = relu(convolve2d(image, kernel))
    fm_norm = normalize_map(fm)

    if pool_mode == 'max':
        pooled, which = max_pool2d(fm)
    else:
        pooled = avg_pool2d(fm)
        which = None

    pooled_norm = normalize_map(pooled)

    return {
        'feature_map': fm_norm.tolist(),
        'fm_shape': list(fm.shape),
        'pooled': pooled_norm.tolist(),
        'pooled_shape': list(pooled.shape),
        'pool_mode': pool_mode,
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  FULL CNN PIPELINE")
    print("=" * 55)
    images = make_images()
    for name, info in images.items():
        r = run_pipeline(info['data'].tolist())
        stages = r['summary']['stages']
        print(f"\n  {info['name']}:")
        for s in stages:
            print(f"    {s['name']:14s}  shape={s['shape']:16s}  params={s['params']}")
