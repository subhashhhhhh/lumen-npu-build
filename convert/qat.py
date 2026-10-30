"""Quantization-aware distillation for an all-8-bit Lumen Cutout (BiRefNet-lite student).

LiteRT's PT2E quantizer only covers convolutions, linear layers, adds, multiplies and concatenation,
so Swin's matmuls, softmax and layer norms would stay float. The NPU build is all-8-bit (int8.py:
every tensor 8-bit, weights 8-bit per channel). So training simulates that instead:

- the NPU graph (birefnet_npu_rewrites + split convolutions + swin_v69, as birefnet_lite.py builds
  it) is captured with torch.export at a fixed batch,
- every float tensor an op produces gets 8-bit fake quantization (per tensor, asymmetric, moving
  min/max), and every convolution / linear weight 8-bit per-channel symmetric (as int8.py),
- the student trains through that rounding (straight-through gradients) against the float teacher.

The result is plain float weights in the original model's names (split convolutions joined back),
so the existing path makes the real model: birefnet_lite.py (float .tflite), then int8.py a8w8
(calibrated on COCO), then compile.yml. Training only makes the weights robust to 8-bit rounding;
int8.py still picks the final scales.
"""
import os, sys
import torch
import torch.nn as nn
from torch.ao.quantization import FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PlainConv(nn.Module):
    """The student's stand-in for a deformable convolution (see birefnet_lite.py)."""
    def __init__(self, deform, dilate):
        super().__init__()
        rc = deform.regular_conv
        k = rc.kernel_size[0]; d = dilate if k == 7 else 1
        self.conv = nn.Conv2d(rc.in_channels, rc.out_channels, rc.kernel_size, padding=d * (k // 2), dilation=d, bias=rc.bias is not None)

    def forward(self, x):
        return self.conv(x)


def plain_student(model, dilate):
    n = 0
    for _, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if type(child).__name__ == "DeformableConv2d":
                setattr(mod, cn, PlainConv(child, dilate)); n += 1
    return n


def npu_graph(model, attention="4d"):
    """The rewrites birefnet_lite.py applies before converting (all exact)."""
    import birefnet_npu_rewrites, swin_v69
    birefnet_npu_rewrites.apply(model, attention)
    birefnet_npu_rewrites.split_wide_convs(model)
    swin_v69.apply(model)
    return model


class Wrapped(nn.Module):
    """Raw RGB 0-255 NHWC in, sigmoid mask NHWC out: what Lumen feeds the .tflite."""
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
        return self.net(x)[-1].sigmoid().permute(0, 2, 3, 1)


class FQ(nn.Module):
    """8-bit fake quantization of an activation: per tensor, asymmetric, moving min/max; in float32
    (the fused op wants it) whatever the input's type."""
    def __init__(self):
        super().__init__()
        self.fq = FusedMovingAvgObsFakeQuantize(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                dtype=torch.qint8, qscheme=torch.per_tensor_affine)

    def forward(self, x):
        # An fp16 overflow must not poison the moving ranges for good.
        y = torch.nan_to_num(x if x.dtype == torch.float32 else x.float(), nan=0.0, posinf=65504.0, neginf=-65504.0)
        return self.fq(y).to(x.dtype)


class WeightFQ(nn.Module):
    """8-bit per-channel symmetric fake quantization of a weight, scaled by each output channel's
    largest magnitude (as int8.py's quantizer does); no observer, the weight itself is the range.
    Gradients pass straight through."""
    def __init__(self):
        super().__init__()
        self.register_buffer("on", torch.tensor(False))

    def forward(self, w):
        if not bool(self.on): return w
        wf = w.float()
        scale = (wf.detach().abs().amax(dim=tuple(range(1, wf.dim()))) / 127).clamp(min=1e-8)
        zero = torch.zeros_like(scale, dtype=torch.int32)
        return torch.fake_quantize_per_channel_affine(wf, scale, zero, 0, -127, 127).to(w.dtype)


# Ops that only move or view data: an int8 graph keeps their input's scale, nothing to round again.
_PASS = {"view", "reshape", "_unsafe_view", "permute", "transpose", "t", "expand", "slice", "squeeze", "unsqueeze",
         "clone", "contiguous", "alias", "select", "unflatten", "flatten", "split", "split_with_sizes", "chunk",
         "unbind", "getitem", "detach", "_to_copy", "to", "lift_fresh_copy", "dropout", "contiguous"}
_WEIGHTED = {"conv2d", "convolution", "linear"}


def _opname(node):
    t = node.target
    name = getattr(t, "__name__", str(t))
    if hasattr(t, "_overloadpacket"): name = t._overloadpacket.__name__
    return name.split(".")[0]


def _is_float(node):
    v = node.meta.get("val")
    return isinstance(v, torch.Tensor) and v.is_floating_point()


def fake_quantize(gm):
    """Puts 8-bit fake quantization after every value-changing op and on every conv/linear weight."""
    g = gm.graph
    acts = weights = 0
    for node in list(g.nodes):
        if node.op == "placeholder" and _is_float(node):
            pass
        elif node.op != "call_function" or not _is_float(node) or _opname(node) in _PASS:
            continue
        name = f"fq_a{acts}"; acts += 1
        gm.add_submodule(name, FQ())
        with g.inserting_after(node):
            q = g.call_module(name, (node,))
        node.replace_all_uses_with(q, delete_user_cb=lambda u: u is not q)
    for node in list(g.nodes):
        if node.op == "call_function" and _opname(node) in _WEIGHTED and len(node.args) > 1:
            w = node.args[1]
            if isinstance(w, torch.fx.Node) and w.op == "get_attr":
                name = f"fq_w{weights}"; weights += 1
                gm.add_submodule(name, WeightFQ())
                with g.inserting_after(w):
                    q = g.call_module(name, (w,))
                node.replace_input_with(w, q)
    g.lint(); gm.recompile()
    return acts, weights


def to_device(gm, dev):
    """.to() moves parameters and buffers; an exported graph also keeps plain tensor constants."""
    gm.to(dev)
    for mod in gm.modules():
        for k, v in list(vars(mod).items()):
            if isinstance(v, torch.Tensor) and not isinstance(v, nn.Parameter) and v.device != torch.device(dev):
                setattr(mod, k, v.to(dev))
    # Ops that make a tensor (arange, zeros, full, scalar_tensor...) carry the export's device.
    target = torch.device(dev); moved = 0
    def swap(a):
        nonlocal moved
        if isinstance(a, torch.device) and a != target:
            moved += 1; return target
        return a
    for node in gm.graph.nodes:
        node.args = torch.fx.node.map_aggregate(node.args, swap)
        node.kwargs = torch.fx.node.map_aggregate(node.kwargs, swap)
    if moved: gm.recompile()
    return gm


def build(student_sd, dilate, side, batch, model_src="ZhengPeng7/BiRefNet_lite", device="cpu"):
    """The rewritten student as a fake-quantized GraphModule (fixed batch and side) on `device`."""
    from transformers import AutoModelForImageSegmentation
    m = AutoModelForImageSegmentation.from_pretrained(model_src, trust_remote_code=True).float().eval()
    plain_student(m, dilate)
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in student_sd.items()}, strict=False)
    assert not unexpected and all(k.startswith("bb.") for k in missing), (unexpected[:3], missing[:3])
    reference_keys = set(m.state_dict())
    npu_graph(m)
    w = Wrapped(m).eval()
    x = torch.rand(batch, side, side, 3) * 255
    # Exported on the CPU (part of the rewritten graph builds CPU tensors), then moved.
    gm = torch.export.export(w, (x,), strict=False).module()
    acts, weights = fake_quantize(gm)
    return to_device(gm, device), reference_keys, (acts, weights)


def fq_modules(gm):
    return [m.fq for m in gm.modules() if isinstance(m, FQ)]


def observe(gm, on):
    for f in fq_modules(gm):
        f.enable_observer() if on else f.disable_observer()


def quantize(gm, on):
    for f in fq_modules(gm):
        f.enable_fake_quant() if on else f.disable_fake_quant()
    for m in gm.modules():
        if isinstance(m, WeightFQ): m.on.fill_(bool(on))


def averaging(gm, c):
    for f in fq_modules(gm):
        f.activation_post_process.averaging_constant = c
        f.averaging_constant = c


def float_weights(gm, reference_keys):
    """The trained weights under the original model's names: split convolutions joined back."""
    sd = {k: v for k, v in gm.state_dict().items() if ".fq." not in k and not k.startswith("fq_")}
    out, parts = {}, {}
    for k, v in sd.items():
        if not k.startswith("net."): continue
        k = k[4:]
        if ".parts." in k:
            base, rest = k.split(".parts.", 1)
            i, field = rest.split(".", 1)
            parts.setdefault((base, field), {})[int(i)] = v
        elif k in reference_keys:
            out[k] = v
    for (base, field), d in parts.items():
        if field == "weight":
            out[f"{base}.weight"] = torch.cat([d[i] for i in sorted(d)], 1)
        elif field == "bias":
            out[f"{base}.bias"] = d[0]
    missing = reference_keys - set(out)
    return out, missing
