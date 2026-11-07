"""M1 validation: can the real 5B pipeline VAE decode per-chunk (streaming)?

Compares 3 decode modes on one random latent sequence using `pipe.vae`:
  full   : decode(z[:, :, 0:T])               (what offline.py does today)
  naive  : decode(z[:, :, a:b]) per chunk, cat (loses causal time context)
  window : decode(z[:, :, a-L : b]) per chunk, drop the re-decoded prefix
           (re-decodes a L-frame lookback so the causal context is present)

Ship the hook whose mode matches FULL closest at chunk boundaries.
"""
import time
import yaml
import torch

from edenlive.core.factory import build_model_adapter, build_runtime_plugins
from edenlive.core.types import RuntimeContext

cfg_path = "edenlive/configs/wan22_5b_720p_few_step.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

ctx = RuntimeContext(device=torch.device("cuda:0"))
model = build_model_adapter(cfg)
for plugin in build_runtime_plugins(cfg):
    ctx = plugin.attach(model, ctx)
model.load_components(ctx)

vae = model.components["pipeline"].vae
vae.eval()
z_dim = vae.config.z_dim
lm = torch.tensor(vae.config.latents_mean, dtype=vae.dtype, device="cuda:0").view(1, z_dim, 1, 1, 1)
ls = torch.tensor(vae.config.latents_std, dtype=vae.dtype, device="cuda:0").view(1, z_dim, 1, 1, 1)

def decode(z):
    z = z / ls + lm
    with torch.no_grad():
        return vae.decode(z, return_dict=True).sample.float().to("cpu")

T, H, W = 7, 12, 12  # 24-frame, 96x96 clip -> 7 latent frames (T=4*6+1)
torch.manual_seed(0)
z = torch.randn(1, z_dim, T, H, W, device="cuda:0", dtype=vae.dtype) * 0.5

torch.cuda.synchronize()
t0 = time.perf_counter()
full = decode(z)
t_full = (time.perf_counter() - t0) * 1000
torch.cuda.empty_cache()
print(f"FULL   : frames={full.shape[2]:3d}  {t_full:6.1f} ms   (reference)")

def report(name, cat, t_ms):
    if cat.shape[2] != full.shape[2]:
        # structural mismatch (causal first-frame rule) — compare the overlapping portion
        n_common = min(cat.shape[2], full.shape[2])
        dmax = (cat[:, :, :n_common] - full[:, :, :n_common]).abs().max().item()
        dmean = (cat[:, :, :n_common] - full[:, :, :n_common]).abs().mean().item()
        rel = dmean / (full[:, :, :n_common].abs().mean().item() + 1e-9) * 100
        print(f"{name:14s} frames={cat.shape[2]:3d}/{full.shape[2]:3d}(common)  {t_ms:6.1f} ms   max|d|={dmax:.4f}  rel={rel:.3f}%   [FRAME-COUNT MISMATCH: structural causal-boundary effect]")
        return
    dmax = (cat - full).abs().max().item()
    dmean = (cat - full).abs().mean().item()
    rel = dmean / (full.abs().mean().item() + 1e-9) * 100
    bnd = []
    for a in range(0, cat.shape[2], 8):
        b = min(a + 8, cat.shape[2])
        bnd.append((cat[:, :, a:b] - full[:, :, a:b]).abs().max().item())
    print(f"{name:14s} frames={cat.shape[2]:3d}  {t_ms:6.1f} ms   max|d|={dmax:.4f}  rel={rel:.3f}%   first-8f max={[f'{x:.3f}' for x in bnd[:4]]}")

for chunk_size in (2, 3, 4):
    n = (T + chunk_size - 1) // chunk_size
    parts, times = [], []
    for i in range(n):
        a, b = i * chunk_size, min((i + 1) * chunk_size, T)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parts.append(decode(z[:, :, a:b]))
        times.append((time.perf_counter() - t0) * 1000)
        torch.cuda.empty_cache()
    cat = torch.cat(parts, dim=2)
    report(f"naive c{chunk_size}", cat, sum(times)); torch.cuda.empty_cache()

print()
L = 2  # latent-frame lookback, re-decoded per chunk
for chunk_size in (2, 3):
    n = (T + chunk_size - 1) // chunk_size
    parts, times = [], []
    for i in range(n):
        a, b = i * chunk_size, min((i + 1) * chunk_size, T)
        a0 = max(0, a - L)
        seg = z[:, :, a0:b]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = decode(seg)
        times.append((time.perf_counter() - t0) * 1000)
        # causal mapping: a segment of n latents decodes to 4n-3 frames, with
        # the first latent -> 1 frame and each subsequent latent -> 4 frames.
        # So the prefix a0..a-1 (a-a0 latents) occupies 4(a-a0)-3 frames, and
        # the target chunk owns the tail 4(b-a) frames.
        if a == a0:
            keep_from = 0
        else:
            keep_from = 4 * (a - a0) - 3
        parts.append(out[:, :, keep_from:])
        torch.cuda.empty_cache()
    cat = torch.cat(parts, dim=2)
    if cat.shape[2] == full.shape[2]:
        report(f"window c{chunk_size} L{L}", cat, sum(times))
    else:
        print(f"window c{chunk_size} L{L}: frame-count mismatch {cat.shape[2]} vs {full.shape[2]} (boundary accounting) — see first-frame rule")
    torch.cuda.empty_cache()

print()
print("Interpretation:")
print("- 'window' row close to FULL + 'naive' drifting at boundaries  -> ship window-style hook")
print("- all naive and window drift similarly  -> VAE decoder is bidirectional in time")
print("  streaming = best-effort preview (log the boundary error), or keep full decode at end")
