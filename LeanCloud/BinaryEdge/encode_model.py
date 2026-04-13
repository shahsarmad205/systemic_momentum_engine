"""
Run locally to encode best_long_model.pkl into model_chunk_XX.py files
that QuantConnect will sync (each under the 64,000 char limit).

Usage:
    python encode_model.py

Output:
    model_chunk_00.py ... model_chunk_NN.py + model_payload.py
    Commit ALL of these to git so QC syncs them.
"""

import base64
import zlib
import os
import sys
import glob

PKL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_long_model.pkl")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
CHUNK_CHARS = 55_000   # well under QC's 64,000 char limit


def encode():
    if not os.path.isfile(PKL_PATH):
        print(f"ERROR: {PKL_PATH} not found")
        sys.exit(1)

    with open(PKL_PATH, "rb") as fh:
        raw = fh.read()

    compressed = zlib.compress(raw, level=9)
    b64 = base64.b64encode(compressed).decode("ascii")

    print(f"Original:   {len(raw):>10,} bytes")
    print(f"Compressed: {len(compressed):>10,} bytes  ({len(compressed)/len(raw)*100:.1f}%)")
    print(f"Base64:     {len(b64):>10,} chars")

    # Remove old chunk files
    for old in glob.glob(os.path.join(OUT_DIR, "model_chunk_??.py")):
        os.remove(old)

    chunks = [b64[i:i + CHUNK_CHARS] for i in range(0, len(b64), CHUNK_CHARS)]
    print(f"\nSplitting into {len(chunks)} chunks...")

    for idx, chunk in enumerate(chunks):
        fname   = os.path.join(OUT_DIR, f"model_chunk_{idx:02d}.py")
        content = (
            f"# AUTO-GENERATED chunk {idx:02d}/{len(chunks)-1} - do not edit.\n"
            f"# Run encode_model.py to regenerate.\n"
            f"DATA = \"{chunk}\"\n"
        )
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"  Written {os.path.basename(fname)}  ({len(content):,} chars)")

    imports = "\n".join(
        f"from model_chunk_{i:02d} import DATA as _c{i:02d}" for i in range(len(chunks))
    )
    joins = " + ".join(f"_c{i:02d}" for i in range(len(chunks)))
    joiner = (
        "# AUTO-GENERATED - do not edit.\n"
        "# Joins model_chunk_XX.py files into the full base64+zlib payload.\n"
        "import base64, zlib, io\n"
        f"{imports}\n"
        f"MODEL_B64_ZLIB = {joins}\n"
        "def load_model_bytes():\n"
        "    return zlib.decompress(base64.b64decode(MODEL_B64_ZLIB))\n"
    )
    with open(os.path.join(OUT_DIR, "model_payload.py"), "w", encoding="utf-8") as fh:
        fh.write(joiner)

    print(f"\nWritten model_payload.py")
    print(f"Commit model_chunk_*.py + model_payload.py to git.")


if __name__ == "__main__":
    encode()
