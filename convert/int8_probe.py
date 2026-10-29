"""Which layers break the 8-bit cutout model: 8-bit activations everywhere except one group of layer
types, which keeps 16-bit activations; checked against the float model on COCO photos.

usage: int8_probe.py <in.tflite> <variant> <photos dir> [calibration count] [check count]
"""
import collections, os, sys, time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer, qtyping, calibrator

Op = qtyping.TFLOperationName
# Layer types given 16-bit activations in each variant (weights stay 8-bit).
VARIANTS = {
    "all8": [],
    "softmax": [Op.SOFTMAX],
    "attention": [Op.SOFTMAX, Op.BATCH_MATMUL],
    "norm": [Op.MEAN, Op.SQUARED_DIFFERENCE, Op.RSQRT, Op.SUB],
    "fc": [Op.FULLY_CONNECTED],
    "gelu": [Op.GELU],
    "elementwise": [Op.ADD, Op.MUL, Op.SUB, Op.DIV],
    "output": [Op.LOGISTIC],
    "elementwise_softmax": [Op.ADD, Op.MUL, Op.SUB, Op.DIV, Op.SOFTMAX],
    # Everything 16-bit except the heavy layers: what an NPU-friendly mix would look like.
    "conv8": "conv8",
}
src, variant, photos = sys.argv[1:4]
calib_n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
check_n = int(sys.argv[5]) if len(sys.argv) > 5 else 20

probe = Interpreter(model_path=src)
signature = next(iter(probe.get_signature_list()))
runner = probe.get_signature_runner(signature)
input_name = next(iter(runner.get_input_details()))
side = runner.get_input_details()[input_name]["shape"][1]
ops = collections.Counter(d["op_name"] for d in probe._get_ops_details())
print("layer types:", ", ".join(f"{k} {v}" for k, v in ops.most_common()), flush=True)
del runner, probe

files = sorted(os.path.join(photos, f) for f in os.listdir(photos) if f.lower().endswith(".jpg"))
calib_files, check_files = files[:calib_n], files[calib_n:calib_n + check_n]


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((side, side), Image.BILINEAR), np.float32)[None]


q = quantizer.Quantizer(src)
if VARIANTS[variant] == "conv8":
    q.add_static_config(regex=".*", operation_name=Op.ALL_SUPPORTED, activation_num_bits=16, weight_num_bits=8)
    for op in (Op.CONV_2D, Op.FULLY_CONNECTED, Op.DEPTHWISE_CONV_2D):
        q.add_static_config(regex=".*", operation_name=op, activation_num_bits=8, weight_num_bits=8)
else:
    q.add_static_config(regex=".*", operation_name=Op.ALL_SUPPORTED, activation_num_bits=8, weight_num_bits=8)
    for op in VARIANTS[variant]:
        q.add_static_config(regex=".*", operation_name=op, activation_num_bits=16, weight_num_bits=8)
start = time.time()
samples = [{input_name: load(f)} for f in calib_files]
calibration = q.calibrate({signature: samples}, num_threads=os.cpu_count(),
                          mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED)
dst = f"probe_{variant}.tflite"
q.quantize(calibration).export_model(dst, overwrite=True)
print(f"{variant}: calibrated and quantized in {time.time() - start:.0f} s, {os.path.getsize(dst) / 1e6:.0f} MB", flush=True)


def masks(path, images):
    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    run = it.get_signature_runner(signature)
    i = run.get_input_details()[input_name]
    o = next(iter(run.get_output_details().values()))
    (i_scale, i_zero), (o_scale, o_zero) = i["quantization"], o["quantization"]
    out, start = [], time.time()
    for image in images:
        if i_scale:
            info = np.iinfo(i["dtype"])
            image = np.clip(np.round(image / i_scale + i_zero), info.min, info.max).astype(i["dtype"])
        mask = next(iter(run(**{input_name: image}).values())).ravel()
        out.append((mask.astype(np.float32) - o_zero) * o_scale if o_scale else mask)
    print(f"{os.path.basename(path)}: {(time.time() - start) / len(images):.1f} s a photo", flush=True)
    return out


images = [load(f) for f in check_files]
want, got = masks(src, images), masks(dst, images)
flips = np.array([((a > .5) != (b > .5)).mean() for a, b in zip(want, got)])
# IoU only where the float model finds a subject (1% of the photo or more): an empty mask has no IoU.
subject = np.array([(a > .5).mean() >= 0.01 for a in want])
iou = np.array([((a > .5) & (b > .5)).sum() / ((a > .5) | (b > .5)).sum() for a, b, k in zip(want, got, subject) if k])
print(f"RESULT {variant}: IoU on {len(iou)} photos with a subject: median {np.median(iou):.4f}, mean {iou.mean():.4f}, worst {iou.min():.3f}; "
      f"pixels changing side median {np.median(flips) * 100:.2f}%, mean {flips.mean() * 100:.2f}% ({len(images)} photos)", flush=True)
