"""Ahead-of-time compiles a .tflite model for one Qualcomm chip with LiteRT's compiler."""
import pathlib, sys
from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.qualcomm import target as qnn


class Soc(str):
    """Any chip the Qualcomm compiler plugin knows, including ones the Python list leaves out (SM7450)."""
    @property
    def value(self): return str(self)


model, soc, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
result = aot_compile.aot_compile(model, output_dir=out, target=qnn.Target(soc_model=Soc(soc)), keep_going=False)
print(result)
for f in sorted(out.iterdir()): print(f.name, f.stat().st_size)
if not any(out.iterdir()): sys.exit("nothing compiled")
