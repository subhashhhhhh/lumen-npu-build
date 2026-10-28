"""Probe models for the Snapdragon 7 Gen 1 (V69, 2 MB VTCM): small pieces of BiRefNet-lite's Swin
backbone, each in a few formulations, to find which ones the Qualcomm compiler can tile.

usage: swin_probe.py <outdir>   (writes <outdir>/probe_<name>.tflite)
"""
import os, sys, importlib.util
import torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import snapshot_download

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
path = snapshot_download("ZhengPeng7/BiRefNet_lite", allow_patterns=["*.py", "*.json"])
def load_birefnet_code(path):
    """BiRefNet's model code as a plain module (its one relative import made absolute)."""
    import types
    sys.path.insert(0, path)
    src = open(os.path.join(path, "birefnet.py")).read().replace("from .BiRefNet_config", "from BiRefNet_config")
    mod = types.ModuleType("birefnet_code"); mod.__file__ = os.path.join(path, "birefnet.py")
    sys.modules["birefnet_code"] = mod; exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


bn = load_birefnet_code(path)
sys.path.insert(0, os.path.dirname(__file__))
import swin_v69
torch.manual_seed(0)
import litert_torch


def export(name, module, *inputs):
    module = module.eval()
    with torch.no_grad():
        module(*inputs)
    litert_torch.convert(module, inputs).export(os.path.join(out, f"probe_{name}.tflite"))
    print("exported", name, flush=True)


class Stage(nn.Module):
    """One Swin stage (a plain block, then a shifted one), tokens in and out."""
    def __init__(self, dim, heads, hw, impl):
        super().__init__()
        self.layer = bn.BasicLayer(dim=dim, depth=2, num_heads=heads, window_size=7)
        self.hw, self.impl = hw, impl

    def forward(self, x):
        if self.impl == "v69":
            return swin_v69.basic_layer_forward(self.layer, x, self.hw, self.hw)[0]
        return self.layer(x, self.hw, self.hw)[0]


class Core(nn.Module):
    """Window attention's core alone: (windows*heads, 49, 32) q, k, v."""
    def __init__(self, split):
        super().__init__(); self.split = split

    def forward(self, q, k, v):
        if self.split <= 1:
            return torch.softmax(q @ k.transpose(-2, -1), -1) @ v
        outs = [torch.softmax(a @ b.transpose(-2, -1), -1) @ c for a, b, c in zip(q.split(self.split), k.split(self.split), v.split(self.split))]
        return torch.cat(outs, 0)


# The original rewrites (as the converter sees them today), applied to the probe's module.
import birefnet_npu_rewrites as rw
bn.window_partition, bn.window_reverse = rw.window_partition, rw.window_reverse
bn.WindowAttention.forward = rw.attention_forward

which = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
probes = {
    "s1_orig": lambda: export("s1_orig", Stage(96, 3, 128, "orig"), torch.randn(1, 128 * 128, 96)),
    "s1_v69": lambda: export("s1_v69", Stage(96, 3, 128, "v69"), torch.randn(1, 128 * 128, 96)),
    "s2_v69": lambda: export("s2_v69", Stage(192, 6, 64, "v69"), torch.randn(1, 64 * 64, 192)),
    "s3_v69": lambda: export("s3_v69", Stage(384, 12, 32, "v69"), torch.randn(1, 32 * 32, 384)),
    "core1083": lambda: export("core1083", Core(1), *[torch.randn(1083, 49, 32) for _ in range(3)]),
    "core1083_split57": lambda: export("core1083_split57", Core(57), *[torch.randn(1083, 49, 32) for _ in range(3)]),
}
for name, make in probes.items():
    if which is None or name in which:
        try:
            make()
        except Exception as e:
            print("FAILED to export", name, repr(e)[:300], flush=True)
