import numpy as np

def relu(xp, x):
    return xp.maximum(x, 0.0)

def flatten(xp, x):
    return xp.reshape(x, (x.shape[0], -1))

def gemm(xp, a, b, c=None):
    y = a @ b.T
    if c is not None:
        y = y + c
    return y

def maxpool2d(xp, x, kernel_shape=(2, 2), strides=(2, 2)):
    n, c, h, w = x.shape
    kh, kw = kernel_shape
    sh, sw = strides

    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1
    out = xp.empty((n, c, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            hs = i * sh
            ws = j * sw
            window = x[:, :, hs:hs + kh, ws:ws + kw]
            out[:, :, i, j] = xp.max(window, axis=(2, 3))

    return out

def conv2d(xp, x, w, b=None, pads=(0, 0, 0, 0), strides=(1, 1)):
    n, cin, h, w_in = x.shape
    cout, cin2, kh, kw = w.shape

    pt, pl, pb, pr = pads
    sh, sw = strides

    xpad = xp.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")

    out_h = ((h + pt + pb - kh) // sh) + 1
    out_w = ((w_in + pl + pr - kw) // sw) + 1
    out = xp.zeros((n, cout, out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            hs = i * sh
            ws = j * sw
            patch = xpad[:, :, hs:hs + kh, ws:ws + kw]
            out[:, :, i, j] = xp.tensordot(
                patch, w, axes=([1, 2, 3], [1, 2, 3])
            )

    if b is not None:
        out += xp.reshape(b, (1, -1, 1, 1))

    return out
