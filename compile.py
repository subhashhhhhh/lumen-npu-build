"""Ahead-of-time compiles a .tflite model for one Qualcomm (SM…) or MediaTek (MT…) chip with LiteRT's compiler."""
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


# The plugin binary's exit status (a crash leaves no message, only a truncated log).
import subprocess
_run = subprocess.run
def _logged_run(*args, **kwargs):
    # Extra compiler flags the Python API doesn't forward, e.g. QUALCOMM_FLAGS="--qualcomm_vtcm_size=2"
    # (an 8 Gen 1 target with a 7 Gen 1's 2 MB of on-chip memory).
    extra = os.environ.get("QUALCOMM_FLAGS", "").split()
    if extra and args and isinstance(args[0], list) and "apply_plugin" in str(args[0][0]):
        args = (args[0] + extra,) + args[1:]
        print("plugin flags:", " ".join(extra), flush=True)
    r = _run(*args, **kwargs)
    if r.returncode: print(f"plugin exit status {r.returncode}", flush=True)
    return r
subprocess.run = _logged_run

model, chip, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
if chip.startswith("MT"):
    from ai_edge_litert.aot.vendors.mediatek import target as mtk
    target = mtk.Target(soc_model=mtk.SocModel(chip))
else:
    target = qnn.Target(soc_model=soc(chip))
try:
    result = aot_compile.aot_compile(model, output_dir=out, target=target, keep_going=False)
except ValueError as e:
    # The plugin's own message is in a file the exception only names: show its end.
    import re
    m = re.search(r"See (\S+\.error)", str(e))
    if m and os.path.exists(m.group(1)):
        import shutil
        shutil.copy(m.group(1), out / (pathlib.Path(model).stem + "_" + chip + ".error"))
        print("---- compiler error ----")
        lines = open(m.group(1), errors="replace").read().splitlines()
        # The cause: errors, else the tiler's "unable to tile" lines (the many tiling.h warnings are noise).
        errors = [l for l in lines if "ERROR" in l or "<E>" in l] or [l for l in lines if "unable to tile" in l or "not sufficiently tiled" in l]
        print("\n".join(l[:400] for l in errors[:40]) or "\n".join(lines[-40:]))
        # The tiler's warnings flood the file; the fatal line is usually among the rest, near the end.
        rest = [l for l in lines if "tiling.h" not in l and "simple_tiler" not in l]
        print("---- compiler log without tiler warnings (last 80) ----")
        print("\n".join(l[:400] for l in rest[-80:]))
        print("---- error file:", len(lines), "lines ----")
    raise
print(result)
for f in sorted(out.iterdir()): print(f.name, f.stat().st_size)
if not any(out.iterdir()): sys.exit("nothing compiled")
