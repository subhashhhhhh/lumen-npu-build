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
        masks = []
        for wy in range(th // ws):
            for wx in range(tw // ws):
                toks = [(kh[wy * ws + i], kw[wx * ws + j]) for i in range(ws) for j in range(ws)]
                m = torch.full((ws * ws, ws * ws), -100.0)
                for a, ta in enumerate(toks):
                    if ta[0] is None or ta[1] is None: continue
                    for b, tb in enumerate(toks):
                        if tb[0] is not None and tb[1] is not None and ta == tb:
                            m[a, b] = 0.0
                masks.append(m)
        _masks[key] = torch.stack(masks)
    return _masks[key]


def block_forward(blk, x, h, w, window_partition, window_reverse, extra_pad=False):
    b, l, c = x.shape
    ws, shift = blk.window_size, blk.shift_size
    shortcut = x
    x = blk.norm1(x).view(b, h, w, c)
    hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
    if shift > 0:
        lead = ws - shift
        x = F.pad(x, (0, 0, lead, wp + ws - lead - w, lead, hp + ws - lead - h))
        mask = shift_mask(h, w, ws, shift).to(x.dtype)
    else:
        # extra_pad: one more row and column of windows, all padding (exact: no real token sees them).
        lead, more = 0, ws if extra_pad else 0
        x = F.pad(x, (0, 0, 0, wp - w + more, 0, hp - h + more))
        mask = None
    _, ph, pw, _ = x.shape
    windows = window_partition(x, ws).reshape(-1, ws * ws, c)
    windows = blk.attn(windows, mask=mask)
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
