"""Restricted loader for PyGOD .pt datasets: only tensor-rebuild functions are allowed; PyG Data is mapped to a plain stub."""
import pickle, zipfile, io, torch, numpy as np
class _Stub:
    def __setstate__(self, s): self.state = s
    def __init__(self, *a, **k): self.state = {}
ALLOW = {("torch._utils", "_rebuild_tensor_v2"), ("collections", "OrderedDict"), ("torch", "FloatStorage"), ("torch", "LongStorage"),
         ("torch", "DoubleStorage"), ("torch", "BoolStorage"), ("torch", "IntStorage"), ("torch", "ByteStorage"), ("torch", "HalfStorage"), ("torch", "ShortStorage"), ("torch", "Size"), ("torch", "float32"), ("torch", "int64")}
def load(path):
    zf = zipfile.ZipFile(path); prefix = zf.namelist()[0].split("/")[0]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) in ALLOW: return getattr(__import__(mod, fromlist=[name]), name)
            if mod.startswith("torch_geometric"): return _Stub
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        def persistent_load(self, pid):
            _, storage_type, key, loc, numel = pid
            raw = zf.read(f"{prefix}/data/{key}")
            dtype = storage_type.dtype if hasattr(storage_type, "dtype") else storage_type(0).dtype
            return torch.storage.TypedStorage(wrap_storage=torch.frombuffer(bytearray(raw), dtype=dtype).untyped_storage(), dtype=dtype, _internal=True)
    obj = U(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    st = obj.state
    def find(d, dep=0):
        if dep > 6: return None
        if isinstance(d, _Stub): return find(d.state, dep + 1)
        if isinstance(d, dict):
            if "x" in d and "edge_index" in d: return d
            for v in d.values():
                r = find(v, dep + 1)
                if r is not None: return r
        if isinstance(d, (list, tuple)):
            for v in d:
                r = find(v, dep + 1)
                if r is not None: return r
    d = find(st)
    return d["x"].numpy(), d["edge_index"].numpy(), d["y"].numpy()
if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        x, e, y = load(p); print(p, x.shape, e.shape, y.shape, y.dtype)
