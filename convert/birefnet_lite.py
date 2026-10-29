"""BiRefNet-lite (ZhengPeng7/BiRefNet_lite, MIT) to LiteRT, for Lumen's long-press cutout.

Its deformable convolutions (torchvision's deform_conv2d: no LiteRT or NPU op) are rebuilt from
grid_sample and a 1x1 convolution; on the owner's photos the masks matched the original to 7e-5,
no pixel changing side. Raw RGB 0-255 in (1x1024x1024x3), subject mask out (1x1024x1024x1, 0-1).

usage: birefnet_lite.py <out.tflite> [side] [student]

Student: a distilled checkpoint (URL or path, decoder-side weights) from the Kaggle distillation. Its
deformable convolutions become plain ones (no gathers at all, which the NPU's tiler can't take).

Side: the square input. 1024 (the model's own) needs bigger NPU tiles than a Snapdragon 7 Gen 1's
TCM takes (a 2.25 MB convolution input slice in the decoder); 768 and 512 are smaller.
"""
import sys
import numpy as np, torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageSegmentation

out = sys.argv[1]
Side = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
path = snapshot_download("ZhengPeng7/BiRefNet_lite")
model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True).float().eval()
Student = sys.argv[3] if len(sys.argv) > 3 else ""
# Dilate: the 7x7 plain convolutions' dilation (v3). The deformable 7x7s sampled 6-10 px off-grid
# (p99); a plain 7x7 reaches 3 px, dilation d reaches 3d. Same weight shape, so v2 loads as is.
Dilate = int(sys.argv[4]) if len(sys.argv) > 4 else 1
if Student:
    class PlainConv(torch.nn.Module):
        def __init__(self, deform):
            super().__init__()
            rc = deform.regular_conv
            k = rc.kernel_size[0]
            d = Dilate if k == 7 else 1
            self.conv = torch.nn.Conv2d(rc.in_channels, rc.out_channels, rc.kernel_size, padding=d * (k // 2), dilation=d, bias=rc.bias is not None)

        def forward(self, x):
            return self.conv(x)
    replaced = 0
    for _, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if type(child).__name__ == "DeformableConv2d":
                setattr(mod, cn, PlainConv(child)); replaced += 1
    if Student.startswith("http"):
        torch.hub.download_url_to_file(Student, "student.pt"); Student = "student.pt"
    sd = torch.load(Student, map_location="cpu")
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and all(k.startswith("bb.") for k in missing), (unexpected[:3], missing[:3])
    model.eval()
    print("student:", replaced, "plain convolutions,", len(sd), "tensors loaded", flush=True)
# Rewritten for Qualcomm's NPU: no deformable convolution or grid_sample, nothing above 4
# dimensions (see birefnet_npu_rewrites.py; checked against the original on the owner's photos).
sys.path.insert(0, __import__("os").path.dirname(__file__))
import birefnet_npu_rewrites
birefnet_npu_rewrites.apply(model)
# The decoder's 960-3072-channel convolutions as input-channel slices (exact): their weights fit the
# Snapdragon 7 Gen 1 / 8 Gen 1 NPU's on-chip memory.
print("split convolutions:", birefnet_npu_rewrites.split_wide_convs(model), flush=True)
# Roll-free shifted windows, and one extra padding window where the 2 MB (7 Gen 1) compiler crashes:
# Swin stage 3's plain block at 16x16, in the half-resolution pass of a 512 input. Both exact.
import swin_v69
swin_v69.apply(model)


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
        return self.net(x)[-1].sigmoid().permute(0, 2, 3, 1)



import faulthandler, time
faulthandler.dump_traceback_later(300, repeat=True)  # a stuck convert shows where it is
import threading, os
_stuck = threading.Timer(1000, lambda: (print('convert stuck: giving up after 1000 s', flush=True), faulthandler.dump_traceback(all_threads=True), os._exit(3)))
_stuck.daemon = True
_stuck.start()
_t0 = time.time()
print('converting', flush=True)
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
print('converted in', round(time.time() - _t0), 's', flush=True)
faulthandler.cancel_dump_traceback_later()
# The phone checks its run against PyTorch's: a test picture and the mask for it (raw float32).
with torch.no_grad():
    want = wrapped(x).numpy()
stem = out.removesuffix(".tflite")
x.numpy().astype("<f4").tofile(f"{stem}_input.bin")
want.astype("<f4").tofile(f"{stem}_reference.bin")
print("reference mask: mean", float(want.mean()), "share above 0.5", float((want > 0.5).mean()), flush=True)
