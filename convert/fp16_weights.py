"""Cutout model with 16-bit float weights (convolutions, fully connected): half the download, still
computed in float. usage: make_fp16.py <in.tflite> <out.tflite> <input.bin> <reference.bin>"""
import sys, os, numpy as np
from ai_edge_quantizer import quantizer, recipe_manager, algorithm_manager, qtyping
src, dst, inp, ref = sys.argv[1:5]
rm = recipe_manager.RecipeManager()
rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, num_bits=16,
                          granularity=qtyping.QuantGranularity.TENSORWISE, algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
q = quantizer.Quantizer(src, rm.get_quantization_recipe())
q.quantize().export_model(dst, overwrite=True)
print(f"{os.path.getsize(src) / 1e6:.0f} MB -> {os.path.getsize(dst) / 1e6:.0f} MB", flush=True)
from ai_edge_litert.interpreter import Interpreter
for path in (src, dst):
    it = Interpreter(model_path=path, num_threads=4); it.allocate_tensors()
    i = it.get_input_details()[0]; it.set_tensor(i["index"], np.fromfile(inp, np.float32).reshape(i["shape"]))
    it.invoke(); out = it.get_tensor(it.get_output_details()[0]["index"]).ravel()
    want = np.fromfile(ref, np.float32)
    print(f"{os.path.basename(path)}: max diff vs PyTorch {np.abs(out - want).max():.5f}, pixels changing side {((out > .5) != (want > .5)).mean() * 100:.3f}%", flush=True)
