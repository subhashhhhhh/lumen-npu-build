"""Ahead-of-time compiles a .tflite model for one Qualcomm chip with LiteRT's compiler."""
import os, pathlib, sys
from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.qualcomm import target as qnn


def soc(name):
    """Any chip the Qualcomm compiler plugin knows, including ones the Python list leaves out (SM7450)."""
    models = qnn.SocModel
    if name not in models._value2member_map_:
        member = str.__new__(models, name)
        member._name_, member._value_ = name, name
        models._member_map_[name] = models._value2member_map_[name] = member
        models._member_names_.append(name)
    return models(name)


model, chip, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
try:
    result = aot_compile.aot_compile(model, output_dir=out, target=qnn.Target(soc_model=soc(chip)), keep_going=False)
except ValueError as e:
    # The plugin's own message is in a file the exception only names: show its end.
    import re
    m = re.search(r"See (\S+\.error)", str(e))
    if m and os.path.exists(m.group(1)):
        print("---- compiler error ----")
        lines = open(m.group(1), errors="replace").read().splitlines()
        # The cause: errors, else the tiler's "unable to tile" lines (the many tiling.h warnings are noise).
        errors = [l for l in lines if "ERROR" in l or "<E>" in l] or [l for l in lines if "unable to tile" in l or "not sufficiently tiled" in l]
        print("\n".join(l[:400] for l in errors[:40]) or "\n".join(lines[-40:]))
    raise
print(result)
for f in sorted(out.iterdir()): print(f.name, f.stat().st_size)
if not any(out.iterdir()): sys.exit("nothing compiled")
