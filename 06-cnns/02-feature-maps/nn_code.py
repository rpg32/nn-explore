"""
=== MODULE 06.2: FEATURE MAPS & FILTERS ===

In Module 06.1 we saw ONE kernel applied to ONE image.
But a real CNN uses MULTIPLE kernels simultaneously!

WHY MULTIPLE KERNELS?
  Each kernel is like a DIFFERENT QUESTION about the image:
    - "Are there horizontal edges here?"     → horizontal edge kernel
    - "Are there vertical edges here?"       → vertical edge kernel
    - "Are there diagonal lines here?"       → diagonal edge kernels
    - "Are there corners here?"              → corner kernel
    - "What's the overall shape?"            → Laplacian kernel

  A single kernel can only detect ONE type of feature.
  To understand a complex image, you need MANY features.

HOW IT WORKS IN A CNN:
  Layer 1 might have 32 kernels → 32 feature maps
  Layer 2 might have 64 kernels → 64 feature maps
  Each layer builds on the previous one's features:
    Layer 1: edges, corners, blobs
    Layer 2: curves, textures, small shapes
    Layer 3: eyes, wheels, letters

THE KEY INSIGHT:
  In a real CNN, these kernels are LEARNED, not hand-designed.
  The network discovers through training which kernels are
  useful for the classification task. We use hand-crafted
  kernels here to build intuition for what learned filters
  actually look for.
"""

import numpy as np


# ============================================================
# CONVOLUTION (from scratch, same as Module 06.1)
# ============================================================
def convolve2d(image, kernel):
    """2D convolution — slide a kernel across an image.

    For each position, compute the weighted sum of the kernel
    and the image patch underneath it. This produces a "feature
    map" showing WHERE that kernel's pattern was detected.

    image:  2D numpy array (H, W)
    kernel: 2D numpy array (kH, kW), typically 3×3

    Returns: feature map of shape (H - kH + 1, W - kW + 1)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    output = np.zeros((out_h, out_w))

    # Slide the kernel across every valid position
    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch under the kernel
            patch = image[i:i + kH, j:j + kW]
            # Weighted sum — exactly what a neuron does!
            output[i, j] = np.sum(patch * kernel)

    return output


# ============================================================
# 8 DIVERSE KERNELS
# Each one "asks a different question" about the image
# ============================================================
KERNELS = [
    {
        'id': 'edge_h',
        'name': 'Horizontal Edge',
        'desc': 'Detects horizontal boundaries — bright above, dark below. '
                'Lights up on the top/bottom edges of shapes.',
        'kernel': [[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]],
    },
    {
        'id': 'edge_v',
        'name': 'Vertical Edge',
        'desc': 'Detects vertical boundaries — bright left, dark right. '
                'Lights up on left/right edges of shapes.',
        'kernel': [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]],
    },
    {
        'id': 'diag_main',
        'name': 'Diagonal \\',
        'desc': 'Detects edges along the top-left to bottom-right diagonal. '
                'Finds slopes and angled lines.',
        'kernel': [[ 0,  1,  1],
                   [-1,  0,  1],
                   [-1, -1,  0]],
    },
    {
        'id': 'diag_anti',
        'name': 'Diagonal /',
        'desc': 'Detects edges along the top-right to bottom-left diagonal. '
                'The opposite angle from the other diagonal kernel.',
        'kernel': [[ 1,  1,  0],
                   [ 1,  0, -1],
                   [ 0, -1, -1]],
    },
    {
        'id': 'corner',
        'name': 'Corner',
        'desc': 'Detects top-left corner patterns — where a horizontal '
                'and vertical edge meet. Strong response at shape corners.',
        'kernel': [[-1, -1,  0],
                   [-1,  4,  0],
                   [ 0,  0,  0]],
    },
    {
        'id': 'blur',
        'name': 'Blur',
        'desc': 'Averages each pixel with its neighbors, smoothing the image. '
                'Removes noise and fine detail.',
        'kernel': [[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]],
    },
    {
        'id': 'sharpen',
        'name': 'Sharpen',
        'desc': 'Enhances edges and detail by amplifying the center pixel '
                'relative to its neighbors. The opposite of blur.',
        'kernel': [[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]],
    },
    {
        'id': 'laplacian',
        'name': 'Laplacian',
        'desc': 'Detects ALL edges regardless of direction. Highlights any '
                'rapid intensity change — outlines the entire shape.',
        'kernel': [[ 0, -1,  0],
                   [-1,  4, -1],
                   [ 0, -1,  0]],
    },
]


# ============================================================
# TEST IMAGES (28×28, generated — no file dependencies)
# ============================================================
def make_test_images():
    """Generate test images that show different kinds of structure.

    Each image has features that different kernels will respond
    to differently — this is exactly the point of having multiple
    kernels!
    """
    images = {}
    sz = 28

    # --- Circle: smooth curved edges in all directions ---
    img = np.zeros((sz, sz))
    cx, cy = sz // 2, sz // 2
    for i in range(sz):
        for j in range(sz):
            if (i - cy) ** 2 + (j - cx) ** 2 < (sz * 0.35) ** 2:
                img[i, j] = 1.0
    images['circle'] = {'name': 'Circle', 'data': img.tolist()}

    # --- Cross / plus sign: pure horizontal + vertical edges ---
    img = np.zeros((sz, sz))
    mid = sz // 2
    img[mid - 2:mid + 2, 4:sz - 4] = 1.0  # horizontal bar
    img[4:sz - 4, mid - 2:mid + 2] = 1.0   # vertical bar
    images['cross'] = {'name': 'Cross', 'data': img.tolist()}

    # --- Digit-like "7": horizontal bar + diagonal stroke ---
    img = np.zeros((sz, sz))
    img[4:7, 6:22] = 1.0  # top horizontal bar
    for i in range(7, 24):
        col = int(22 - (i - 7) * 0.8)
        img[i, max(0, col - 1):min(sz, col + 2)] = 1.0
    images['seven'] = {'name': 'Digit "7"', 'data': img.tolist()}

    # --- Letter "L": vertical stroke + horizontal base ---
    img = np.zeros((sz, sz))
    img[4:22, 8:11] = 1.0   # vertical stroke
    img[19:22, 8:20] = 1.0  # horizontal base
    images['letter_l'] = {'name': 'Letter "L"', 'data': img.tolist()}

    # --- Checkerboard: maximum edges in all directions ---
    img = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(sz):
            if ((i // 4) + (j // 4)) % 2 == 0:
                img[i, j] = 1.0
    images['checker'] = {'name': 'Checkerboard', 'data': img.tolist()}

    return images


# ============================================================
# APPLY ALL KERNELS TO ONE IMAGE
# This is the main function: one image in, 8 feature maps out
# ============================================================
def apply_all_kernels(image_name, images_dict):
    """Apply all 8 kernels to a single image.

    This simulates what the first convolutional layer of a CNN does:
    take ONE input image and produce MULTIPLE feature maps, one per
    learned filter.

    Returns a list of results, one per kernel, each containing:
      - kernel info (name, description, values)
      - the raw feature map
      - a normalized (0-1) display version
      - statistics (min, max)
    """
    if image_name not in images_dict:
        return []

    image = np.array(images_dict[image_name]['data'])
    results = []

    for kinfo in KERNELS:
        kernel = np.array(kinfo['kernel'])
        # Apply convolution
        feature_map = convolve2d(image, kernel)

        # Normalize for display (map to 0-1 range)
        fm_display = feature_map.copy()
        if fm_display.max() != fm_display.min():
            fm_display = (fm_display - fm_display.min()) / (fm_display.max() - fm_display.min())
        else:
            fm_display = np.zeros_like(fm_display)

        results.append({
            'id': kinfo['id'],
            'name': kinfo['name'],
            'desc': kinfo['desc'],
            'kernel': kinfo['kernel'],
            'feature_map': feature_map.tolist(),
            'feature_map_display': fm_display.tolist(),
            'fm_min': float(feature_map.min()),
            'fm_max': float(feature_map.max()),
            'fm_shape': list(feature_map.shape),
        })

    return results


# ============================================================
# CLI sanity check
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  FEATURE MAPS & FILTERS")
    print("  Each kernel asks a different question about the image")
    print("=" * 55)

    images = make_test_images()
    for img_name, img_info in images.items():
        print(f"\n  Image: {img_info['name']}")
        results = apply_all_kernels(img_name, images)
        for r in results:
            print(f"    {r['name']:18s}: output {r['fm_shape']}  "
                  f"range [{r['fm_min']:7.2f}, {r['fm_max']:7.2f}]")
