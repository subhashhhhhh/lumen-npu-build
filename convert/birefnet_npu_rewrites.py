"""BiRefNet rewritten for Qualcomm's NPU: the same maths with no tensor above 4 dimensions and no
deformable-convolution or grid_sample op. Each function replaces one in BiRefNet's own code
(apply(model) swaps them in); convert.py checks the result against the original on real photos.
"""
import sys
import torch
import torch.nn.functional as F


def deformable_forward(self, x):
    """Modulated deformable convolution: for each kernel tap k at (i, j), x sampled bilinearly at
    (h - pad + i + dy, w - pad + j + dx), zero outside (as torchvision), times the modulator; taps x
    channels then go through one 1x1 convolution with the regular kernel's weights. The sampling is
    gathers on the flattened image (3 dimensions at most)."""
    n, c, h, w = x.shape
    weight = self.regular_conv.weight
    kh, kw = weight.shape[2], weight.shape[3]
    k = kh * kw
    pad = self.padding if isinstance(self.padding, int) else self.padding[0]
    offset = self.offset_conv(x)                               # (n, 2K, h, w): dy, dx per tap
    modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))    # (n, K, h, w)
    dy = offset[:, 0::2]                                       # (n, K, h, w)
    dx = offset[:, 1::2]
    taps = torch.arange(k, dtype=x.dtype).view(1, k, 1, 1)
    ti = torch.div(taps, kw, rounding_mode="floor")
    tj = taps - ti * kw
    py = torch.arange(h, dtype=x.dtype).view(1, 1, h, 1) - pad + ti + dy
    px = torch.arange(w, dtype=x.dtype).view(1, 1, 1, w) - pad + tj + dx
    y0 = torch.floor(py)
    x0 = torch.floor(px)
    ly = py - y0
    lx = px - x0
    flat = x.reshape(n, c, h * w)
    points = k * h * w
    sampled = None
    for cy, wy in ((y0, 1 - ly), (y0 + 1, ly)):
        for cx, wx in ((x0, 1 - lx), (x0 + 1, lx)):
            inside = ((cy >= 0) & (cy <= h - 1) & (cx >= 0) & (cx <= w - 1)).to(x.dtype)
            index = (cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1)).to(torch.int64).reshape(n, 1, points)
            corner = torch.gather(flat, 2, index.expand(n, c, points))           # (n, c, K*h*w)
            part = corner * (wy * wx * inside).reshape(n, 1, points)
            sampled = part if sampled is None else sampled + part
    sampled = sampled * modulator.reshape(n, 1, points)
    stacked = sampled.reshape(n, c * k, h, w)                                    # channel-major, tap-minor
    return F.conv2d(stacked, weight.reshape(weight.shape[0], c * k, 1, 1), self.regular_conv.bias)


def window_partition(x, window_size):
    """Swin's windows, in the same order, in 4 dimensions."""
    b, h, w, c = x.shape
    x = x.reshape(b * (h // window_size), window_size, w // window_size, window_size * c)
    return x.permute(0, 2, 1, 3).reshape(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    c = windows.shape[-1]
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.reshape(b * (h // window_size), w // window_size, window_size, window_size * c)
    return x.permute(0, 2, 1, 3).reshape(b, h, w, c)


def attention_forward(self, x, mask=None):
    """Swin window attention with q, k and v split along the last axis (no 5-D reshape) and the
    shift mask added per window in 4 dimensions."""
    b_, n, c = x.shape
    heads = self.num_heads
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = q.reshape(b_, n, heads, c // heads).permute(0, 2, 1, 3) * self.scale
    k = k.reshape(b_, n, heads, c // heads).permute(0, 2, 1, 3)
    v = v.reshape(b_, n, heads, c // heads).permute(0, 2, 1, 3)
    attn = q @ k.transpose(-2, -1)
    bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
    attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
    if mask is not None:
        attn = attn + mask.repeat(b_ // mask.shape[0], 1, 1).unsqueeze(1)
    attn = self.softmax(attn)
    x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
    return self.proj_drop(self.proj(x))


def image2patches(image, grid_h=2, grid_w=2, patch_ref=None, transformation=None):
    """'b c (hg h) (wg w) -> b (c hg wg) h w' (the decoder's only use) in 4 dimensions."""
    if patch_ref is not None:
        grid_h, grid_w = image.shape[-2] // patch_ref.shape[-2], image.shape[-1] // patch_ref.shape[-1]
    assert transformation in (None, "b c (hg h) (wg w) -> b (c hg wg) h w"), transformation
    b, c, hh, ww = image.shape
    ph, pw = hh // grid_h, ww // grid_w
    x = image.reshape(b * c * grid_h, ph, grid_w, pw).permute(0, 2, 1, 3)        # (b c hg, wg, h, w)
    return x.reshape(b, c * grid_h * grid_w, ph, pw)


def apply(model):
    module = sys.modules[type(model).__module__]
    assert not getattr(module.config, "SDPA_enabled", False) if hasattr(module, "config") else True
    module.window_partition = window_partition
    module.window_reverse = window_reverse
    module.image2patches = image2patches
    type(next(m for m in model.modules() if type(m).__name__ == "WindowAttention")).forward = attention_forward
    type(next(m for m in model.modules() if type(m).__name__ == "DeformableConv2d")).forward = deformable_forward
