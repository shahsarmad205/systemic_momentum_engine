# AUTO-GENERATED - do not edit.
# Joins model_chunk_XX.py files into the full base64+zlib payload.
import base64, zlib, io
from model_chunk_00 import DATA as _c00
from model_chunk_01 import DATA as _c01
from model_chunk_02 import DATA as _c02
from model_chunk_03 import DATA as _c03
from model_chunk_04 import DATA as _c04
from model_chunk_05 import DATA as _c05
from model_chunk_06 import DATA as _c06
from model_chunk_07 import DATA as _c07
from model_chunk_08 import DATA as _c08
from model_chunk_09 import DATA as _c09
from model_chunk_10 import DATA as _c10
from model_chunk_11 import DATA as _c11
MODEL_B64_ZLIB = _c00 + _c01 + _c02 + _c03 + _c04 + _c05 + _c06 + _c07 + _c08 + _c09 + _c10 + _c11
def load_model_bytes():
    return zlib.decompress(base64.b64decode(MODEL_B64_ZLIB))
