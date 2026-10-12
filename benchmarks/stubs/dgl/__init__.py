# Stub: AnomalyGFM zero-shot uses dgl only for dgl.random.seed and an unused helper.
class _R:
    @staticmethod
    def seed(s): pass
random = _R()
def DGLGraph(*a, **k): raise RuntimeError("dgl stub")
