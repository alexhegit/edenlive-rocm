import torch
import numpy as np
from decord import VideoReader

base = "inference"
for name in ("cryo-crossing", "pulse-skirmish"):
    p_bidir = f"{base}/wan22_5b_720p_bidir/save/inference/base/{name}.mp4"
    p_fs = f"{base}/wan22_5b_720p_few_step/save/inference/base/{name}.mp4"
    a = VideoReader(p_bidir)
    n = len(a)
    h, w = a[0].shape[:2]
    print(f"=== {name} ===")
    print(f"bidir: {n} frames, {w}x{h}")
    idx = torch.linspace(0, n - 1, 8).long().tolist()
    arr = a.get_batch(idx).asnumpy()
    prev = None
    for k, i in enumerate(idx):
        g = arr[k].astype("float32") / 255.0
        extra = ""
        if prev is not None:
            extra = f" frame_diff={float(np.abs(g - prev).mean()):.4f}"
        print(
            f"  f={i:3d} std={float(g.std()):.3f} "
            f"ch_std={[round(float(g[..., c].std()), 3) for c in range(3)]}{extra}"
        )
        prev = g
    try:
        b = VideoReader(p_fs)
        m = min(len(a), len(b))
        fa = a.get_batch([m // 2]).asnumpy()[0].astype("float32") / 255.0
        fb = b.get_batch([m // 2]).asnumpy()[0].astype("float32") / 255.0
        print(f"vs few_step (mid frame): bidir std={float(fa.std()):.3f} "
              f"fewstep std={float(fb.std()):.3f} "
              f"|diff|={float((fa - fb).abs().mean()):.4f}")
    except Exception as e:
        print(f"(few_step comparison skipped: {e})")
