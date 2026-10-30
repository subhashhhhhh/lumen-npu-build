"""Split-resolution BiRefNet: the Swin backbone on a small copy of the photo, the decoder's image
references (BiRefNet's "bilateral reference") from the full-size photo, and the mask at full size.

The backbone is ~84% of the NPU time and grows with its input's pixels; the decoder is ~8%. So the
backbone sees `bb_side` (e.g. 512) while the model takes and returns `side` (e.g. 1024).

mode "tail":    decoder stages 4..1 take their image patches from the backbone-size photo (exactly the
                trained model's inputs, no new weights); only the last stage (ipt_blk1 + conv_out1)
                works on the full-size photo, at full size.
mode "patches": every decoder stage takes its image patches from the full-size photo: (side/bb_side)^2
                times the input channels for ipt_blk5..2. New first convolutions, initialised so each
                full-size tile's channels share the old weight (divided by the count): needs training.
Eval only (no multi-scale supervision or gradient labels), as converted for the phone.
"""
import types
import torch
import torch.nn.functional as F


def _decoder_forward(self, features, img_stage, img_final):
    import sys
    patches = sys.modules[__name__].image2patches
    x1, x2, x3, x4 = features
    cfg = self.config

    def ipt(blk, img, ref, size):
        p = patches(img, patch_ref=ref) if self.split else img
        return blk(F.interpolate(p, size=size, mode="bilinear", align_corners=True))

    x4 = torch.cat((x4, ipt(self.ipt_blk5, img_stage, x4, x4.shape[2:])), 1)
    p4 = self.decoder_block4(x4)
    if cfg.out_ref:
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
    _p3 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True) + self.lateral_block4(x3)
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
    _p1 = F.interpolate(_p1, size=img_final.shape[2:], mode="bilinear", align_corners=True)
    _p1 = torch.cat((_p1, ipt(self.ipt_blk1, img_final, _p1, img_final.shape[2:])), 1)
    return [self.conv_out1(_p1)]


def _widen(blk, k2):
    """ipt block's first conv for k2 times the patch channels (full-size tiles)."""
    c = blk.conv1
    new = torch.nn.Conv2d(c.in_channels * k2, c.out_channels, c.kernel_size, c.stride, c.padding, bias=c.bias is not None)
    with torch.no_grad():
        new.weight.copy_(c.weight.repeat_interleave(k2, dim=1) / k2)
        if c.bias is not None: new.bias.copy_(c.bias)
    blk.conv1 = new


def apply(model, side, bb_side, mode):
    """Call after loading weights and before the NPU rewrites (split_wide_convs sees the widened convs)."""
    assert mode in ("tail", "patches"), mode
    import sys
    rewrites = sys.modules.get("birefnet_npu_rewrites")
    globals()["image2patches"] = rewrites.image2patches if rewrites else sys.modules[type(model).__module__].image2patches
    dec = model.decoder
    if mode == "patches":
        assert side % bb_side == 0, ("patches mode needs side to be a multiple of bb_side", side, bb_side)
        k2 = (side // bb_side) ** 2
        for n in ("ipt_blk5", "ipt_blk4", "ipt_blk3", "ipt_blk2"):
            _widen(getattr(dec, n), k2)
    dec.forward = types.MethodType(_decoder_forward, dec)

    def forward(self, x):
        xb = F.avg_pool2d(x, side // bb_side) if side % bb_side == 0 else \
            F.interpolate(x, size=(bb_side, bb_side), mode="bilinear", align_corners=False, antialias=False)
        (x1, x2, x3, x4), _ = self.forward_enc(xb)
        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        return self.decoder([x1, x2, x3, x4], x if mode == "patches" else xb, x)
    model.forward = types.MethodType(forward, model)
    return model
