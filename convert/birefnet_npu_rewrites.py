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
    channels then go through one 1x1 convolution with the regular kernel's weights. The sampling
    gathers whole rows (one pixel's channels) from the image laid out pixel-major: a plain Gather
    the NPU compiler can tile, at most 3 dimensions. Batch of one (the app's only use)."""
    n, c, h, w = x.shape
    assert n == 1
    weight = self.regular_conv.weight
    kh, kw = weight.shape[2], weight.shape[3]
    k = kh * kw
    pad = self.padding if isinstance(self.padding, int) else self.padding[0]
    offset = self.offset_conv(x)                               # (1, 2K, h, w): dy, dx per tap
    modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))    # (1, K, h, w)
    dy = offset[:, 0::2]
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
    rows = x.reshape(c, h * w).t()                             # (h*w, c): one row per pixel
    points = k * h * w
    sampled = None
    for cy, wy in ((y0, 1 - ly), (y0 + 1, ly)):
        for cx, wx in ((x0, 1 - lx), (x0 + 1, lx)):
            inside = ((cy >= 0) & (cy <= h - 1) & (cx >= 0) & (cx <= w - 1)).to(x.dtype)
            index = (cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1)).to(torch.int32).reshape(points)
            part = rows.index_select(0, index) * (wy * wx * inside).reshape(points, 1)   # (K*h*w, c)
            sampled = part if sampled is None else sampled + part
    sampled = sampled * modulator.reshape(points, 1)
    stacked = sampled.reshape(k, h * w, c).permute(2, 0, 1).reshape(1, c * k, h, w)  # channel-major, tap-minor
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


def attention_forward(self, x, mask=None, rpi=None):
    """Swin window attention with q, k and v split along the last axis (no 5-D reshape) and the
    shift mask added per window in 4 dimensions."""
    b_, n, c = x.shape
    heads = self.num_heads
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = q.reshape(b_, n, heads, c // heads).permute(0, 2, 1, 3) * self.scale
    k = k.reshape(b_, n, heads, c // heads).permute(0, 2, 1, 3)
    v = v.reshape(b_, n, heads, c // heads).permute(0, 2, 1, 3)
    attn = q @ k.transpose(-2, -1)
    bias = self.relative_position_bias_table[(self.relative_position_index if rpi is None else rpi).view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
    attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
    if mask is not None:
        attn = attn + mask.repeat(b_ // mask.shape[0], 1, 1).unsqueeze(1)
    attn = self.softmax(attn)
    x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
    return self.proj_drop(self.proj(x))


def attention_forward_heads(self, x, mask=None, rpi=None):
    """The same window attention one head at a time, in 3 dimensions: no reshape or transpose of
    q, k, v or the attention map (on the 7 Gen 1 NPU those layout shuffles cost about a fifth of
    the whole model at int8). Heads are channel slices of qkv, as in the original; the outputs are
    concatenated in head order. Batch of one adds no reshape at all (the app's case); bigger batches
    (training) reshape only for the shifted-window mask."""
    b_, n, c = x.shape
    heads, d = self.num_heads, c // self.num_heads
    qkv = self.qkv(x)                                                  # (b_, n, 3c)
    bias = self.relative_position_bias_table[(self.relative_position_index if rpi is None else rpi).view(-1)].view(n, n, heads)
    nw = mask.shape[0] if mask is not None else b_
    outs = []
    for h in range(heads):
        q = qkv[:, :, h * d:(h + 1) * d] * self.scale
        k = qkv[:, :, c + h * d:c + (h + 1) * d]
        v = qkv[:, :, 2 * c + h * d:2 * c + (h + 1) * d]
        attn = torch.bmm(q, k.transpose(1, 2)) + bias[:, :, h].unsqueeze(0)
        if mask is not None:
            if nw == b_:  # batch of one: the phone's graph, no reshape
                attn = attn + mask
            else:  # training batches: the same mask for every photo
                attn = (attn.view(b_ // nw, nw, n, n) + mask.unsqueeze(0)).view(b_, n, n)
        outs.append(torch.bmm(self.softmax(attn), v))
    return self.proj_drop(self.proj(torch.cat(outs, dim=-1)))


def image2patches(image, grid_h=2, grid_w=2, patch_ref=None, transformation=None):
    """'b c (hg h) (wg w) -> b (c hg wg) h w' (the decoder's only use) in 4 dimensions."""
    if patch_ref is not None:
        grid_h, grid_w = image.shape[-2] // patch_ref.shape[-2], image.shape[-1] // patch_ref.shape[-1]
    assert transformation in (None, "b c (hg h) (wg w) -> b (c hg wg) h w"), transformation
    b, c, hh, ww = image.shape
    ph, pw = hh // grid_h, ww // grid_w
    x = image.reshape(b * c * grid_h, ph, grid_w, pw).permute(0, 2, 1, 3)        # (b c hg, wg, h, w)
    return x.reshape(b, c * grid_h * grid_w, ph, pw)


class SplitConv(torch.nn.Module):
    """A convolution with very many input channels as a sum of convolutions over slices of them.
    Same result; each part's weights fit the Snapdragon 7 Gen 1's 2 MB on-chip memory."""
    def __init__(self, conv, chunk):
        super().__init__()
        self.parts = torch.nn.ModuleList()
        for i, a in enumerate(range(0, conv.in_channels, chunk)):
            b = min(a + chunk, conv.in_channels)
            part = torch.nn.Conv2d(b - a, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                                   bias=conv.bias is not None and i == 0)
            part.weight.data.copy_(conv.weight.data[:, a:b])
            if part.bias is not None: part.bias.data.copy_(conv.bias.data)
            self.parts.append(part)
        self.bounds = [(a, min(a + chunk, conv.in_channels)) for a in range(0, conv.in_channels, chunk)]

    def forward(self, x):
        out = None
        for (a, b), part in zip(self.bounds, self.parts):
            y = part(x[:, a:b])
            out = y if out is None else out + y
        return out


def split_wide_convs(model, budget=6912):
    """Splits every ungrouped convolution whose kernel area x input channels exceeds `budget`
    (3x3 over more than 768 channels): at 32 output channels a slice, its fp16 weights alone
    would take ~0.45 MB or more of on-chip memory."""
    n = 0
    for _, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if type(child) is torch.nn.Conv2d and child.groups == 1:
                k2 = child.kernel_size[0] * child.kernel_size[1]
                if k2 * child.in_channels > budget:
                    setattr(mod, cn, SplitConv(child, max(32, budget // k2 // 32 * 32))); n += 1
    return n


def apply(model, attention="4d"):
    """attention: "4d" (heads batched in 4 dimensions) or "heads" (one head at a time, no layout
    shuffles). Both exact."""
    module = sys.modules[type(model).__module__]
    assert not getattr(module.config, "SDPA_enabled", False) if hasattr(module, "config") else True
    module.window_partition = window_partition
    module.window_reverse = window_reverse
    module.image2patches = image2patches
    type(next(m for m in model.modules() if type(m).__name__ == "WindowAttention")).forward = (
        {"4d": attention_forward, "heads": attention_forward_heads}[attention])
    # The distilled student has no deformable convolutions left (plain ones instead): nothing to swap.
    deform = next((m for m in model.modules() if type(m).__name__ == "DeformableConv2d"), None)
    if deform is not None:
        type(deform).forward = deformable_forward
