"""BiRefNet-lite (ZhengPeng7/BiRefNet_lite, MIT) to LiteRT, for Lumen's long-press cutout.

Its deformable convolutions (torchvision's deform_conv2d: no LiteRT or NPU op) are rebuilt from
grid_sample and a 1x1 convolution; on the owner's photos the masks matched the original to 7e-5,
no pixel changing side. Raw RGB 0-255 in (1x1024x1024x3), subject mask out (1x1024x1024x1, 0-1).

usage: birefnet_lite.py <out.tflite>
"""
import sys
import numpy as np, torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageSegmentation

out = sys.argv[1]
Side = 1024
path = snapshot_download("ZhengPeng7/BiRefNet_lite")
model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True).float().eval()
DeformableConv2d = type(next(m for m in model.modules() if type(m).__name__ == "DeformableConv2d"))


def plain_forward(self, x):
    """Modulated deformable convolution from standard ops. For each kernel tap k at (i, j): sample x
    at (h - pad + i + dy, w - pad + j + dx), bilinear, zero outside, times the modulator; then all
    taps x channels through one 1x1 convolution with the regular kernel's weights."""
    n, c, h, w = x.shape
    weight = self.regular_conv.weight
    kh, kw = weight.shape[2], weight.shape[3]
    pad = self.padding if isinstance(self.padding, int) else self.padding[0]
    offset = self.offset_conv(x)                              # (n, 2K, h, w): dy, dx per tap
    modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))  # (n, K, h, w)
    ys = torch.arange(h, dtype=x.dtype).view(1, h, 1)
    xs = torch.arange(w, dtype=x.dtype).view(1, 1, w)
    taps = []
    for k in range(kh * kw):
        i, j = divmod(k, kw)
        py = ys - pad + i + offset[:, 2 * k]
        px = xs - pad + j + offset[:, 2 * k + 1]
        # align_corners=True: pixel centres at -1 and +1 for the first and last row/column.
        grid = torch.stack((px * (2.0 / (w - 1)) - 1.0, py * (2.0 / (h - 1)) - 1.0), dim=-1)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        taps.append(sampled * modulator[:, k:k + 1])
    stacked = torch.stack(taps, dim=2).reshape(n, c * kh * kw, h, w)  # channel-major, tap-minor
    out = F.conv2d(stacked, weight.reshape(weight.shape[0], c * kh * kw, 1, 1), self.regular_conv.bias)
    return out



DeformableConv2d.forward = plain_forward


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
        return self.net(x)[-1].sigmoid().permute(0, 2, 3, 1)



import litert_torch
wrapped = Wrapped(model).eval()
rng = np.random.default_rng(0)
# Smooth synthetic pictures (blurred noise), not real photos: those stay on the owner's machines.
def picture():
    x = torch.from_numpy(rng.uniform(0, 255, (1, 3, 64, 64)).astype(np.float32))
    return F.interpolate(x, size=(Side, Side), mode="bicubic").clamp(0, 255).permute(0, 2, 3, 1).contiguous()
edge = litert_torch.convert(wrapped, (picture(),))
edge.export(out)
for _ in range(2):
    x = picture()
    with torch.no_grad():
        want = wrapped(x).numpy()
    got = edge(x.numpy())
    print(f"tflite vs torch: max diff {np.abs(got - want).max():.5f}", flush=True)
