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
    """Modulated deformable convolution from standard ops, in one sampling step: the sampling
    points of all K kernel taps side by side in one grid (h x w*K), bilinear, zero outside, times
    the modulator; then taps x channels through one 1x1 convolution with the regular kernel's
    weights. (One grid_sample per tap made a graph the converter and compiler choked on.)"""
    n, c, h, w = x.shape
    weight = self.regular_conv.weight
    kh, kw = weight.shape[2], weight.shape[3]
    k_taps = kh * kw
    pad = self.padding if isinstance(self.padding, int) else self.padding[0]
    offset = self.offset_conv(x).reshape(n, k_taps, 2, h, w)   # dy, dx per tap
    modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))     # (n, K, h, w)
    ti = (torch.arange(k_taps, dtype=x.dtype) // kw).view(1, k_taps, 1, 1)
    tj = (torch.arange(k_taps, dtype=x.dtype) % kw).view(1, k_taps, 1, 1)
    ys = torch.arange(h, dtype=x.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, dtype=x.dtype).view(1, 1, 1, w)
    py = ys - pad + ti + offset[:, :, 0]                        # (n, K, h, w)
    px = xs - pad + tj + offset[:, :, 1]
    # align_corners=True: pixel centres at -1 and +1 for the first and last row/column.
    gx = px * (2.0 / (w - 1)) - 1.0
    gy = py * (2.0 / (h - 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).permute(0, 2, 3, 1, 4).reshape(n, h, w * k_taps, 2)  # (n, h, w*K, 2), tap-minor
    sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (n, c, h, w*K)
    sampled = sampled.reshape(n, c, h, w, k_taps).permute(0, 1, 4, 2, 3)                          # (n, c, K, h, w)
    sampled = sampled * modulator.unsqueeze(1)
    stacked = sampled.reshape(n, c * k_taps, h, w)               # channel-major, tap-minor
    return F.conv2d(stacked, weight.reshape(weight.shape[0], c * k_taps, 1, 1), self.regular_conv.bias)



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
x = picture()
edge = litert_torch.convert(wrapped, (x,))
edge.export(out)
# The phone checks its run against PyTorch's: a test picture and the mask for it (raw float32).
with torch.no_grad():
    want = wrapped(x).numpy()
stem = out.removesuffix(".tflite")
x.numpy().astype("<f4").tofile(f"{stem}_input.bin")
want.astype("<f4").tofile(f"{stem}_reference.bin")
print("reference mask: mean", float(want.mean()), "share above 0.5", float((want > 0.5).mean()), flush=True)
