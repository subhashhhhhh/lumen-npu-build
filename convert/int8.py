"""Cutout model with integer weights and activations, for NPUs: calibrated on COCO photos, then
checked against the float model on other COCO photos.

usage: int8.py <in.tflite> <out.tflite> <scheme> <photos dir> [calibration count] [check count]
scheme: a8w8 (8-bit activations and weights) or a16w8 (16-bit activations, 8-bit weights)
"""
import os, sys, time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer, qtyping, calibrator

src, dst, scheme, photos = sys.argv[1:5]
calib_n = int(sys.argv[5]) if len(sys.argv) > 5 else 64
check_n = int(sys.argv[6]) if len(sys.argv) > 6 else 100
bits = {"a8w8": 8, "a16w8": 16}[scheme]

probe = Interpreter(model_path=src)
signature = next(iter(probe.get_signature_list()))
runner = probe.get_signature_runner(signature)
input_name = next(iter(runner.get_input_details()))
side = runner.get_input_details()[input_name]["shape"][1]
del runner, probe

files = sorted(os.path.join(photos, f) for f in os.listdir(photos) if f.lower().endswith(".jpg"))
calib_files, check_files = files[:calib_n], files[calib_n:calib_n + check_n]


def load(path):
    # What Lumen feeds it: the photo stretched to a square, raw RGB 0-255.
    return np.asarray(Image.open(path).convert("RGB").resize((side, side), Image.BILINEAR), np.float32)[None]


q = quantizer.Quantizer(src)
q.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                    activation_num_bits=bits, weight_num_bits=8)
start = time.time()
# A list, not a generator: the calibrator walks the data more than once.
samples = [{input_name: load(f)} for f in calib_files]
print(f"{len(files)} photos found; calibrating on {len(samples)}", flush=True)
calibration = q.calibrate({signature: samples}, num_threads=os.cpu_count(),
                          mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED)
print(f"calibrated on {len(calib_files)} photos in {time.time() - start:.0f} s", flush=True)
q.quantize(calibration).export_model(dst, overwrite=True)
print(f"{os.path.getsize(src) / 1e6:.0f} MB -> {os.path.getsize(dst) / 1e6:.0f} MB", flush=True)


def masks(path, images):
    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    run = it.get_signature_runner(signature)
    # A quantized model takes and gives integers: scale the photo in and the mask out.
    i = run.get_input_details()[input_name]
    o = next(iter(run.get_output_details().values()))
    (i_scale, i_zero), (o_scale, o_zero) = i["quantization"], o["quantization"]
    print(f"{os.path.basename(path)}: input {i['dtype'].__name__} {i['quantization']}, output {o['dtype'].__name__} {o['quantization']}", flush=True)
    out, start = [], time.time()
    for image in images:
        if i_scale:
            info = np.iinfo(i["dtype"])
            image = np.clip(np.round(image / i_scale + i_zero), info.min, info.max).astype(i["dtype"])
        mask = next(iter(run(**{input_name: image}).values())).ravel()
        out.append((mask.astype(np.float32) - o_zero) * o_scale if o_scale else mask)
    print(f"{os.path.basename(path)}: {(time.time() - start) / len(images):.1f} s a photo on this CPU", flush=True)
    return out


images = [load(f) for f in check_files]
want, got = masks(src, images), masks(dst, images)
flips = np.array([((a > .5) != (b > .5)).mean() for a, b in zip(want, got)])
diff = np.array([np.abs(a - b).mean() for a, b in zip(want, got)])
# IoU only where the float model finds a subject (1% of the photo or more): an empty mask has no IoU.
subject = np.array([(a > .5).mean() >= 0.01 for a in want])
iou = np.array([((a > .5) & (b > .5)).sum() / ((a > .5) | (b > .5)).sum() for a, b, k in zip(want, got, subject) if k])
print(f"{scheme} vs float on {len(images)} COCO photos: pixels changing side mean {flips.mean() * 100:.3f}%, "
      f"median {np.median(flips) * 100:.3f}%, worst {flips.max() * 100:.2f}%; mean diff {diff.mean():.4f}; "
      f"IoU on {len(iou)} photos with a subject: median {np.median(iou):.4f}, worst {iou.min():.3f}", flush=True)
