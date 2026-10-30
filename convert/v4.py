"""Lumen Cutout v4 graph: split resolution (backbone at bb_side, mask at side) plus speed changes.

Options (a comma list; "" = plain split-resolution tail mode, the same as hybrid.py's):
  two     The backbone's photo is a second input (the app resizes it on the GPU) instead of a
          non-integer resize inside the graph. Exact apart from the resize itself.
  slim    The full-size last stage, thinned: the decoder's 96 features squeezed to 8 at low
          resolution before going up to full size (a 1x1 convolution commutes with bilinear
          upsampling), and a 3->16->8 image block instead of 3->64->24. Initialised so channel 0
          carries the old output's decoder part exactly; the image part starts at zero. Needs training.
  global4 Swin stage 4 (20x20 at 640, 10x10 in the half pass) as one window, no shift: no padding
          windows. Relative position bias extended from 13x13 offsets to 39x39 (edge values
          repeated outward). Needs training.
  nohalf  No half-resolution backbone pass (BiRefNet's multi-scale input, about a fifth of the
          backbone): its features replaced by the full pass's own. Needs training.
  prune4  No Swin stage 4, squeeze block or decoder stage 4. In every distilled student so far
          (v3, QAT r1, r2) that branch's attention gate collapsed to ~1e-11, so it is dead compute:
          dropping it changes the mask by under 1e-6. Exact for those students.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

SLIM_P, SLIM_I = 8, 8  # squeezed decoder channels, image-block channels at full size


def _decoder_forward(self, features, img_stage, img_final):
    import sys
    patches = sys.modules[__name__].image2patches
    x1, x2, x3, x4 = features
    cfg = self.config

    def ipt(blk, img, ref, size):
        p = patches(img, patch_ref=ref) if self.split else img
        return blk(F.interpolate(p, size=size, mode="bilinear", align_corners=True))

    if x4 is None:  # prune4: the dead stage-4 branch contributes zero
        _p3 = self.lateral_block4(x3)
    else:
        x4 = torch.cat((x4, ipt(self.ipt_blk5, img_stage, x4, x4.shape[2:])), 1)
        p4 = self.decoder_block4(x4)
        if cfg.out_ref:
            p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p3 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True) + self.lateral_block4(x3)
    aux = p4 if (self.aux and x4 is not None) else None
    _p3 = torch.cat((_p3, ipt(self.ipt_blk4, img_stage, _p3, x3.shape[2:])), 1)
    p3 = self.decoder_block3(_p3)
    if cfg.out_ref:
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
    _p2 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True) + self.lateral_block3(x2)
    _p2 = torch.cat((_p2, ipt(self.ipt_blk3, img_stage, _p2, x2.shape[2:])), 1)
    p2 = self.decoder_block2(_p2)
    if cfg.out_ref:
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
    _p1 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True) + self.lateral_block2(x1)
    _p1 = torch.cat((_p1, ipt(self.ipt_blk2, img_stage, _p1, x1.shape[2:])), 1)
    _p1 = self.decoder_block1(_p1)
    if self.slim:
        _p1 = self.tail_squeeze(_p1)  # at low resolution: the expensive part never sees 96 channels
    _p1 = F.interpolate(_p1, size=img_final.shape[2:], mode="bilinear", align_corners=True)
    _p1 = torch.cat((_p1, ipt(self.ipt_blk1, img_final, _p1, img_final.shape[2:])), 1)
    return [self.conv_out1(_p1)] if aux is None else [aux, self.conv_out1(_p1)]


def _slim(dec):
    old = dec.conv_out1[0]                       # Conv2d(96 + 24, 1, 1)
    cp = old.in_channels - dec.ipt_blk1.conv_out.out_channels
    sq = nn.Conv2d(cp, SLIM_P, 1, bias=False)
    nn.init.normal_(sq.weight, std=0.02)
    out = nn.Conv2d(SLIM_P + SLIM_I, 1, 1)
    with torch.no_grad():
        sq.weight[0] = old.weight[0, :cp]        # channel 0 = the old decoder contribution
        out.weight.zero_(); out.weight[0, 0] = 1.0
        out.bias.copy_(old.bias)
    blk = dec.ipt_blk1
    blk.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
    blk.conv_out = nn.Conv2d(16, SLIM_I, 3, 1, 1)
    dec.tail_squeeze = sq
    dec.conv_out1 = nn.Sequential(out)


def _global4(model, bb_side, nohalf):
    """Stage 4 as one window: bias table for offsets up to +-(W-1), W = stage 4's side (20 at 640);
    the index for each map size it meets (full pass, half pass) as non-saved buffers."""
    layer = model.bb.layers[3]
    W = bb_side // 32
    sizes = [W] if nohalf else [W, W // 2]
    for blk in layer.blocks:
        a = blk.attn
        t = a.relative_position_bias_table.data              # (13*13, heads)
        w0 = a.window_size[0]
        g = t.t().reshape(1, -1, 2 * w0 - 1, 2 * w0 - 1)
        pad = W - w0
        g = F.pad(g, (pad, pad, pad, pad), mode="replicate")   # far offsets take the edge values
        a.relative_position_bias_table = nn.Parameter(g.reshape(-1, (2 * W - 1) ** 2).t().contiguous())
        a.glob_w = W
        for n in sizes:
            a.register_buffer(f"gidx{n}", global_index(n, W), persistent=False)
        blk.glob = True


def global_index(n, W, device=None):
    c = torch.stack(torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")).flatten(1)
    r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (W - 1)
    return (r[..., 0] * (2 * W - 1) + r[..., 1]).to(device)


def _bb_forward3(self, x):
    """Swin without stage 4 (and without the patch merging that feeds it)."""
    x = self.patch_embed(x)
    wh, ww = x.size(2), x.size(3)
    x = self.pos_drop(x.flatten(2).transpose(1, 2))
    outs = []
    for i in range(3):
        x_out, h, w, x, wh, ww = self.layers[i](x, wh, ww)
        outs.append(getattr(self, f"norm{i}")(x_out).view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous())
    return tuple(outs)


def _forward_enc_v4(self, x):
    """The encoder with nohalf and/or prune4 (same as the original otherwise)."""
    full = self.bb(x)
    if self.v4_nohalf:
        half = full
    else:
        h, w = x.shape[2:]
        half = [F.interpolate(t, size=f.shape[2:], mode="bilinear", align_corners=True) for t, f in
                zip(self.bb(F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)), full)]
    mode = self.config.mul_scl_ipt
    feats = [torch.cat([f, g], 1) if mode == "cat" else f + g for f, g in zip(full, half)]
    if self.v4_prune4:
        return (*feats, None), None
    x1, x2, x3, x4 = feats
    if self.config.cxt:
        x4 = torch.cat((*[F.interpolate(t, size=x4.shape[2:], mode="bilinear", align_corners=True)
                          for t in (x1, x2, x3)][-len(self.config.cxt):], x4), dim=1)
    return (x1, x2, x3, x4), None


def apply(model, side, bb_side, options="", aux=False):
    """aux: the decoder also returns stage 4's gated features first (training: feature loss, gate watch)."""
    """After loading weights, before the NPU rewrites' split_wide_convs and swin_v69 (which reads
    blk.glob). Returns the option set."""
    import sys
    opts = set(o for o in options.split(",") if o)
    assert opts <= {"two", "slim", "global4", "nohalf", "prune4"}, opts
    assert not {"global4", "prune4"} <= opts, "global4 and prune4 exclude each other"
    import birefnet_npu_rewrites as rewrites
    globals()["image2patches"] = rewrites.image2patches
    dec = model.decoder
    dec.aux = aux
    dec.slim = "slim" in opts
    if dec.slim:
        _slim(dec)
    dec.forward = types.MethodType(_decoder_forward, dec)
    if "global4" in opts:
        _global4(model, bb_side, "nohalf" in opts)
    model.v4_nohalf, model.v4_prune4 = "nohalf" in opts, "prune4" in opts
    if model.v4_prune4:
        model.bb.forward = types.MethodType(_bb_forward3, model.bb)
        model.bb.layers[2].downsample = None
    if model.v4_nohalf or model.v4_prune4:
        model.forward_enc = types.MethodType(_forward_enc_v4, model)

    def forward(self, x, xb=None):
        if xb is None:
            xb = x if bb_side == side else (F.avg_pool2d(x, side // bb_side) if side % bb_side == 0 else
                 F.interpolate(x, size=(bb_side, bb_side), mode="bilinear", align_corners=False))
        (x1, x2, x3, x4), _ = self.forward_enc(xb)
        if self.config.squeeze_block and x4 is not None:
            x4 = self.squeeze_module(x4)
        return self.decoder([x1, x2, x3, x4], xb, x)
    model.forward = types.MethodType(forward, model)
    model.v4_two = "two" in opts
    return opts


class Wrapped(nn.Module):
    """Raw RGB 0-255 NHWC in (one input, or the full-size and backbone-size photos), mask NHWC out.
    aux: (mask, stage 4's gated features) for training."""
    def __init__(self, net, aux=False):
        super().__init__(); self.net = net; self.aux = aux
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x, xb=None):
        n = lambda t: (t.permute(0, 3, 1, 2) - self.mean) / self.std
        out = self.net(n(x), None if xb is None else n(xb))
        mask = out[-1].sigmoid().permute(0, 2, 3, 1)
        return (mask, out[0]) if self.aux else mask


BRANCH = ("squeeze_module.", "decoder.decoder_block4.", "decoder.gdt_convs_4.", "decoder.gdt_convs_attn_4.",
          "decoder.ipt_blk5.", "bb.layers.3.", "bb.norm3.")


def revive_stage4(student, teacher):
    """The dead stage-4 branch back from the teacher (call before apply). Deformable convolutions map
    to the student's plain ones through their regular_conv weights (offsets dropped)."""
    tsd, n = teacher.state_dict(), 0
    with torch.no_grad():
        for k, v in student.state_dict().items():
            if not k.startswith(BRANCH):
                continue
            tk = k if k in tsd else k.replace(".conv.weight", ".regular_conv.weight").replace(".conv.bias", ".regular_conv.bias")
            if tk in tsd and tsd[tk].shape == v.shape:
                v.copy_(tsd[tk]); n += 1
    return n
