"""P2 bench: bf16 vs Int8WeightOnly vs Int8W8A8 vs FP8W8A8 on MI300X (torchao 0.18).

Focus: does a real *GEMM speedup* appear (not just weight savings)?
W8A8 = activation+weight both int8/fp8 → int8/fp8 GEMM hardware path.
"""
import time
import torch
import torch.nn as nn

torch.manual_seed(0)
dev = "cuda:0"
H, F = 4096, 14336
x = torch.randn(1, 2048, H, device=dev, dtype=torch.bfloat16)

def bench(fn, iters=20):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

base = nn.Linear(H, F, bias=False).to(dev).to(torch.bfloat16)
base_w = base.weight.detach().clone()
ref = base(x)
t_bf16 = bench(lambda xx: base(xx))
print(f"bf16 baseline: {t_bf16:.3f} ms/GEMM  (anchor for speedup)")

from torchao.quantization import quantize_
import torchao.quantization.quant_api as qa

def make(name, cfg_cls, **kw):
    m = nn.Linear(H, F, bias=False).to(dev).to(torch.bfloat16)
    m.load_state_dict({"weight": base_w})
    return m, cfg_cls(**kw)

def try_cfg(label, cfg_cls, use_compile, **kw):
    m, cfg = make(label, cfg_cls, **kw)
    try:
        quantize_(m, cfg)
    except Exception as e:
        print(f"{label:38s} quantize_ FAIL: {repr(e)[:140]}")
        return
    fn = (lambda t: torch.compile(t, backend="inductor", fullgraph=False))(m) if use_compile else m
    try:
        y = fn(x)
    except Exception as e:
        print(f"{label:38s} forward FAIL: {repr(e)[:140]}")
        return
    err = (y.float() - ref.float()).abs().mean().item()
    rel = err / (ref.float().abs().mean().item() + 1e-6)
    cs = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.float().flatten(), dim=0).item()
    try:
        t = bench(lambda xx: fn(xx))
        print(f"{label:38s} cs={cs:.5f} rel={rel*100:.2f}%  {t:.3f} ms  ({t_bf16/t:.2f}x vs bf16)")
    except Exception as e:
        print(f"{label:38s} cs={cs:.5f} rel={rel*100:.2f}%  (bench failed: {repr(e)[:80]})")

print()
for use_compile in (False, True):
    tag = "+compile" if use_compile else "eager   "
    print(f"--- {tag} ---")
    try_cfg(f"Int8WeightOnly      ({tag})", qa.Int8WeightOnlyConfig, use_compile)
    try_cfg(f"Int8W8A8-dynamicAct ({tag})", qa.Int8DynamicActivationInt8WeightConfig, use_compile)
    try_cfg(f"FP8W8A8-dynamicAct  ({tag})", qa.Float8DynamicActivationFloat8WeightConfig, use_compile)
print()
print(f"anchor bf16: {t_bf16:.3f} ms")
