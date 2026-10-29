---
license: mit
base_model: ZhengPeng7/BiRefNet_lite
pipeline_tag: image-segmentation
library_name: litert
tags:
  - background-removal
  - dichotomous-image-segmentation
  - birefnet
  - litert
  - tflite
  - on-device
  - android
  - distillation
datasets:
  - detection-datasets/coco
---

# Lumen Cutout: BiRefNet-lite for phones

The model behind "touch and hold to cut out" in [Lumen](https://github.com/subhashhhhhh/Lumen), a
local-first Android gallery. It lifts the subject of a photo off its background, on the phone.

This is [BiRefNet-lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) (Zheng et al.) changed so
that phone GPUs and NPUs can run it, then distilled from the original to win back its accuracy.

## Why a changed model

BiRefNet's decoder uses **deformable convolutions**. Neither LiteRT's GPU path nor Qualcomm's NPU
compiler supports them, and rebuilding them from standard ops (`grid_sample`) creates gathers the
NPU can't tile. So the 20 deformable convolutions are replaced with **plain convolutions** of the
same size, started from the original's regular kernels.

That alone flips about 10% of mask pixels. Distillation brings it back:

- **Teacher:** the original BiRefNet-lite. **Student:** the plain-convolution version.
- **Data:** [COCO 2017](https://cocodataset.org) train images (118k), no labels: the student
  learns to reproduce the teacher's masks. Mostly at 512×512, some at 768×768, with augmentation.
- **What trains:** the decoder and squeeze module. The Swin backbone stays frozen (the original's).
- **Loss:** BCE + L1 against the teacher's mask, with 5× weight on a band around its edges.
- **Optimizer:** AdamW, lr 1e-4, mixed precision. Two runs on one Kaggle T4.

## Results

Pixels whose side of 0.5 differs from the original BiRefNet-lite (lower is better):

| | Before distillation | After |
|---|---|---|
| COCO val (200 images), 512 | 10.4% | **3.84%** |
| COCO val, 768 | | 4.13% |
| The Lumen developer's own photos (180, private, not used for training), 512: median | | **0.38%** |

On the developer's photos the median overlap (IoU) with the original is 0.99, and 94% of photos
come out above 0.9. COCO's cluttered scenes are harder than typical phone photos.

**Known weakness:** busy scenes. Without deformable convolutions the decoder sees less context, and
on a few photos (5 of 180) it takes a large part of the background for the subject. A next
version with larger kernels is planned.

## Files

| File | |
|---|---|
| `cutout_512.tflite` | LiteRT model, 134 MB. Weights of convolutions and fully connected layers in float16, computed in float32 (or float16 on GPU). |
| `pytorch/decoder_v2.pt` | The distilled weights for PyTorch (everything but the frozen backbone, float16), 29 MB. |

**`cutout_512.tflite`**
- Input: `1×512×512×3` float32, **raw RGB 0–255** (NHWC). The photo stretched to a square; the
  ImageNet normalization is inside the model.
- Output: `1×512×512×1` float32, the subject mask, **0–1** (the sigmoid is inside the model).
  Stretch it back over the photo.

Speed on a Snapdragon 7 Gen 1 (Motorola Edge 50): **160 ms on the GPU** (LiteRT, float16), 1.8 s on
four CPU cores. The first GPU load builds the GPU's programs (~10 s); keep them with LiteRT's
`serializationDir` and later loads take ~1.5 s.

It also compiles ahead of time for Qualcomm NPUs (LiteRT 2.3, QAIRT 2.50) on 9 chips: Snapdragon
7 Gen 1, 8 Gen 1, 8+ Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite and 8 Elite Gen 5 variants. On the 7 Gen 1 the
NPU took 559 ms, slower than the GPU, so Lumen uses the GPU.

## PyTorch

```python
import torch
from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download

model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)

class PlainConv(torch.nn.Module):
    def __init__(self, deform):
        super().__init__()
        rc = deform.regular_conv
        self.conv = torch.nn.Conv2d(rc.in_channels, rc.out_channels, rc.kernel_size,
                                    padding=deform.padding, bias=rc.bias is not None)
    def forward(self, x):
        return self.conv(x)

for module in list(model.modules()):
    for name, child in list(module.named_children()):
        if type(child).__name__ == "DeformableConv2d":
            setattr(module, name, PlainConv(child))

weights = torch.load(hf_hub_download("subhashhhhhh/lumen-cutout", "pytorch/decoder_v2.pt"))
model.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)  # the backbone stays the original's
model.eval()
# Input: RGB 0-1, ImageNet-normalized, 512x512. Mask: model(x)[-1].sigmoid()
```

## License and credit

MIT, like BiRefNet. Please credit the original:

```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume={3},
  pages={9150038},
  year={2024}
}
```

Distillation used COCO 2017 images ([COCO terms of use](https://cocodataset.org/#termsofuse)). No
COCO images or labels are included here.
