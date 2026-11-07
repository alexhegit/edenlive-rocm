"""P1 probe: does ROCm inductor (torch.compile) actually compile+run on this GPU?

Compiles a representative DiT-block-style subgraph (RMSNorm + GEMM + SDPA + RoPE)
with backend="inductor" and checks (a) it runs, (b) it's numerically ~= eager,
(c) it's faster than eager. This tells us whether turning runtime.torch_compile
on is worth the full-run comparison.
"""
import time, torch, torch.nn.functional as F

dev = "cuda:0"
torch.manual_seed(0)

# shapes ~ one DiT self-attention block, 5B Wan2.2: batch1, seq ~3k, heads 24, head_dim 128
B, S, H, D = 1, 3000, 24, 128
x = torch.randn(B, S, H * D, device=dev, dtype=torch.bfloat16)
w = torch.randn(H * D, H * D, device=dev, dtype=torch.bfloat16) * 0.02
norm_w = torch.ones(H * D, device=dev, dtype=torch.bfloat16)

def block_eager(x):
    h = F.rms_norm(x, (H * D,), weight=norm_w)
    h = h @ w
    q = h.unflatten(-1, (H, D)).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, q, q)
    out = o.transpose(1, 2).flatten(2, 3)
    return out @ w

# -- eager baseline (time it, warm)
for _ in range(2):
    y_eager = block_eager(x)
t0 = time.perf_counter(); 
for _ in range(3):
    y_eager = block_eager(x)
torch.cuda.synchronize(); t_eager = (time.perf_counter() - t0) / 3

# -- inductor compile
print("compiling with inductor (backend)...", flush=True)
t0 = time.perf_counter()
fn = torch.compile(block_eager, backend="inductor", fullgraph=False)
y_ind = fn(x)                      # first call triggers compile
torch.cuda.synchronize()
t_compile = time.perf_counter() - t0
print(f"compile+warmup wall: {t_compile:.1f}s", flush=True)

t0 = time.perf_counter()
for _ in range(3):
    y_ind = fn(x)
torch.cuda.synchronize(); t_ind = (time.perf_counter() - t0) / 3

diff = (y_ind.float() - y_eager.float()).abs().max().item()
print(f"eager  : {t_eager * 1e3:.2f} ms / block")
print(f"inductor: {t_ind * 1e3:.2f} ms / block  ({t_eager / t_ind:.2f}x)")
print(f"max|inductor-eager| = {diff:.5f} (bf16, expect ~1e-2)")
print("OK inductor compiles+runs on ROCm/MI300X" if diff < 0.1 else "!! numeric mismatch")
