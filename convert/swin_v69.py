"""Swin stages for the Snapdragon 7 Gen 1 NPU (V69, 2 MB VTCM).

The shifted-window blocks' torch.roll becomes slices and concatenations along height and width,
which this compiler can't tile at stage-1 size. The same result without any roll: pad the map by
(window - shift) on the top and left, partition into aligned windows, and use a fixed mask that
reproduces the original's windows and shift regions exactly. One extra row and column of windows
(400 instead of 361 at stage 1), no roll, no concatenation.
"""
import torch
import torch.nn.functional as F


def _keys(n_real, n_pad, ws, shift):
    """For each row of the new padded map: (valid, original window, original shift region)."""
    lead = ws - shift
    total = n_pad + ws                      # n_pad (the original's padded size) is a multiple of ws
    keys = []
    for p in range(total):
        r = p - lead
        if not 0 <= r < n_pad:
            keys.append(None); continue
        s = (r - shift) % n_pad             # where the original's roll put this row
        region = 0 if s < n_pad - ws else (1 if s < n_pad - shift else 2)
        keys.append((s // ws, region))
    return keys, lead, total


_masks = {}


def shift_mask(h, w, ws, shift):
    """(windows, ws*ws, ws*ws) additive mask, -100 where the original wouldn't attend."""
    key = (h, w, ws, shift)
    if key not in _masks:
        hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
        kh, _, th = _keys(h, hp, ws, shift)
        kw, _, tw = _keys(w, wp, ws, shift)
        # Built in numpy: torch ops here would be traced one element at a time by torch.export
        # (millions of nodes at 512, an export that never finishes).
        import numpy as np
        def ids(keys):  # each row/column's key as an integer, -1 for padding
            u = {k: i for i, k in enumerate(dict.fromkeys(k for k in keys if k is not None))}
            return np.array([-1 if k is None else u[k] for k in keys])
        ih, iw = ids(kh), ids(kw)
        nh, nw = th // ws, tw // ws
        tok_h = ih.reshape(nh, ws)[:, None, :, None].repeat(nw, 1).repeat(ws, 3)   # (nh, nw, ws, ws)
        tok_w = iw.reshape(nw, ws)[None, :, None, :].repeat(nh, 0).repeat(ws, 2)
        valid = (tok_h >= 0) & (tok_w >= 0)
        tok = np.where(valid, tok_h * (iw.max() + 2) + tok_w, -1).reshape(nh * nw, ws * ws)
        same = (tok[:, :, None] == tok[:, None, :]) & (tok[:, :, None] >= 0)
        _masks[key] = np.where(same, 0.0, -100.0).astype(np.float32)
    return torch.from_numpy(_masks[key])


def block_forward(blk, x, h, w, window_partition, window_reverse, extra_pad=False):
    b, l, c = x.shape
    ws, shift = blk.window_size, blk.shift_size
    rpi = None
    if getattr(blk, "glob", False):  # v4 global4: the whole (small) map is one window, no shift
        ws, shift = max(h, w), 0
        rpi = getattr(blk.attn, f"gidx{ws}")  # registered by v4.apply for this size
        extra_pad = False
    shortcut = x
    x = blk.norm1(x).view(b, h, w, c)
    hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
    if shift > 0:
        lead = ws - shift
        x = F.pad(x, (0, 0, lead, wp + ws - lead - w, lead, hp + ws - lead - h))
        mask = shift_mask(h, w, ws, shift).to(device=x.device, dtype=x.dtype)
    else:
        # extra_pad: one more row and column of windows, all padding (exact: no real token sees them).
        lead, more = 0, ws if extra_pad else 0
        x = F.pad(x, (0, 0, 0, wp - w + more, 0, hp - h + more))
        mask = None
    _, ph, pw, _ = x.shape
    windows = window_partition(x, ws).reshape(-1, ws * ws, c)
    windows = blk.attn(windows, mask=mask) if rpi is None else blk.attn(windows, mask=mask, rpi=rpi)
    x = window_reverse(windows.reshape(-1, ws, ws, c), ws, ph, pw)
    x = x[:, lead:lead + h, lead:lead + w, :].reshape(b, h * w, c)
    x = shortcut + blk.drop_path(x)
    return x + blk.drop_path(blk.mlp(blk.norm2(x)))


def basic_layer_forward(layer, x, h, w, window_partition=None, window_reverse=None, extra_pad=False):
    """BasicLayer.forward with the roll-free shifted blocks."""
    import sys
    mod = sys.modules[type(layer).__module__]
    wp_, wr_ = window_partition or mod.window_partition, window_reverse or mod.window_reverse
    for blk in layer.blocks:
        blk.H, blk.W = h, w
        x = block_forward(blk, x, h, w, wp_, wr_, extra_pad)
    if layer.downsample is not None:
        return x, h, w, layer.downsample(x, h, w), (h + 1) // 2, (w + 1) // 2
    return x, h, w, x, h, w


def needs_extra_pad(h, w, c):
    """Where the 2 MB (Snapdragon 7 Gen 1) compiler crashes: a plain block of Swin-T's stage 3 at
    16x16 (the half-resolution pass of a 512 input), padded to 3x3 windows. One more row and column
    of padding windows (exact) avoids it."""
    return (h, w, c) == (16, 16, 384)


def apply(model):
    """Swaps BasicLayer.forward for the roll-free version, with the extra padding where needed."""
    lay = next(m for m in model.modules() if type(m).__name__ == "BasicLayer")
    type(lay).forward = lambda self, x, H, W: basic_layer_forward(self, x, H, W, extra_pad=needs_extra_pad(H, W, x.shape[-1]))
