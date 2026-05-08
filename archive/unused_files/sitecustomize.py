
import atexit, json, sys
from pathlib import Path

ROOT = Path('/Users/sarmad/Desktop/Quant-project/trend_signal_engine').resolve()
LOG = Path('/Users/sarmad/Desktop/Quant-project/trend_signal_engine/.unused_audit_tracer/runtime_imports.jsonl').resolve()
SEEN = set()

def profiler(frame, event, arg):
    if event != "call":
        return profiler
    filename = frame.f_code.co_filename
    if not filename:
        return profiler
    try:
        p = Path(filename).resolve()
        if p.suffix == ".py" and ROOT in p.parents:
            SEEN.add(str(p))
    except Exception:
        pass
    return profiler

def dump():
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as fh:
        for item in sorted(SEEN):
            fh.write(json.dumps({"file": item}) + "\n")

sys.setprofile(profiler)
atexit.register(dump)
