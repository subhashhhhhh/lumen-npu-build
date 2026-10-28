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


def full_model(v69, plain=True, split=False):
    """BiRefNet-lite with the NPU rewrites (and the distilled student's plain convolutions)."""
    from transformers import AutoModelForImageSegmentation
    m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True).float().eval()
    if plain:
        for _, mod in list(m.named_modules()):
            for cn, child in list(mod.named_children()):
                if type(child).__name__ == "DeformableConv2d":
                    rc = child.regular_conv
                    setattr(mod, cn, nn.Conv2d(rc.in_channels, rc.out_channels, rc.kernel_size, padding=child.padding, bias=rc.bias is not None))
    rw.apply(m)
    if split:
        print("split convolutions:", rw.split_wide_convs(m), flush=True)
    if v69:
        lay = next(x for x in m.modules() if type(x).__name__ == "BasicLayer")
        type(lay).forward = lambda self, x, H, W: swin_v69.basic_layer_forward(self, x, H, W)
    return m


class Backbone(nn.Module):
    def __init__(self, m):
        super().__init__(); self.bb = m.bb

    def forward(self, x):
        return self.bb(x)


class Full(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x)[-1]

class Cut(nn.Module):
    """The model up to a point: 'sq' (encoder + squeeze), 'p4'..'p2' (after that decoder stage),
    'p1' (decoder_block1, before the last upsampling), 'out' (everything)."""
    def __init__(self, m, stop):
        super().__init__(); self.m, self.stop = m, stop

    def forward(self, x):
        m, d = self.m, self.m.decoder
        mod = sys.modules[type(m).__module__]
        (x1, x2, x3, x4), _ = m.forward_enc(x)
        if m.config.squeeze_block:
            x4 = m.squeeze_module(x4)
        if self.stop == "sq": return x4

        def ipt(blk, ref, size):
            pb = mod.image2patches(x, patch_ref=ref, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if d.split else x
            return blk(F.interpolate(pb, size=size, mode='bilinear', align_corners=True))

        def stage(p, block, gdt, attn):
            p = block(p)
            if d.config.out_ref:
                p = p * attn(gdt(p)).sigmoid()
            return p

        if d.config.dec_ipt: x4 = torch.cat((x4, ipt(d.ipt_blk5, x4, x4.shape[2:])), 1)
        p4 = stage(x4, d.decoder_block4, d.gdt_convs_4, d.gdt_convs_attn_4)
        if self.stop == "p4": return p4
        p3 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True) + d.lateral_block4(x3)
        if d.config.dec_ipt: p3 = torch.cat((p3, ipt(d.ipt_blk4, p3, x3.shape[2:])), 1)
        p3 = stage(p3, d.decoder_block3, d.gdt_convs_3, d.gdt_convs_attn_3)
        if self.stop == "p3": return p3
        p2 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True) + d.lateral_block3(x2)
        if d.config.dec_ipt: p2 = torch.cat((p2, ipt(d.ipt_blk3, p2, x2.shape[2:])), 1)
        p2 = stage(p2, d.decoder_block2, d.gdt_convs_2, d.gdt_convs_attn_2)
        if self.stop == "p2": return p2
        p1 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True) + d.lateral_block2(x1)
        if d.config.dec_ipt: p1 = torch.cat((p1, ipt(d.ipt_blk2, p1, x1.shape[2:])), 1)
        p1 = d.decoder_block1(p1)
        if self.stop == "p1": return p1
        p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        if d.config.dec_ipt: p1 = torch.cat((p1, ipt(d.ipt_blk1, p1, x.shape[2:])), 1)
        return d.conv_out1(p1)


probes = {
    **{f"cut256_{s}": (lambda s=s: export(f"cut256_{s}", Cut(full_model(False, split=True), s), torch.randn(1, 3, 256, 256)))
       for s in ("sq", "p4", "p3", "p2", "p1", "out")},
    "s1_orig": lambda: export("s1_orig", Stage(96, 3, 128, "orig"), torch.randn(1, 128 * 128, 96)),
    "s1_v69": lambda: export("s1_v69", Stage(96, 3, 128, "v69"), torch.randn(1, 128 * 128, 96)),
    "s2_v69": lambda: export("s2_v69", Stage(192, 6, 64, "v69"), torch.randn(1, 64 * 64, 192)),
    "s3_v69": lambda: export("s3_v69", Stage(384, 12, 32, "v69"), torch.randn(1, 32 * 32, 384)),
    "core1083": lambda: export("core1083", Core(1), *[torch.randn(1083, 49, 32) for _ in range(3)]),
    "core1083_split57": lambda: export("core1083_split57", Core(57), *[torch.randn(1083, 49, 32) for _ in range(3)]),
    "bb512": lambda: export("bb512", Backbone(full_model(False)), torch.randn(1, 3, 512, 512)),
    "full256": lambda: export("full256", Full(full_model(False)), torch.randn(1, 3, 256, 256)),
    "full384": lambda: export("full384", Full(full_model(False)), torch.randn(1, 3, 384, 384)),
    "full256_split": lambda: export("full256_split", Full(full_model(False, split=True)), torch.randn(1, 3, 256, 256)),
    "full512_split": lambda: export("full512_split", Full(full_model(False, split=True)), torch.randn(1, 3, 512, 512)),
    # The v69 ones patch BasicLayer for good, so they come last.
    "bb512_v69": lambda: export("bb512_v69", Backbone(full_model(True)), torch.randn(1, 3, 512, 512)),
    "full512_v69_split": lambda: export("full512_v69_split", Full(full_model(True, split=True)), torch.randn(1, 3, 512, 512)),
    "full512_v69": lambda: export("full512_v69", Full(full_model(True)), torch.randn(1, 3, 512, 512)),
}
for name, make in probes.items():
    if which is None or name in which:
        try:
            make()
        except Exception as e:
            print("FAILED to export", name, repr(e)[:300], flush=True)
