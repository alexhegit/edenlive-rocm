"""Sanity: SDPA block-attention fallback matches a naive masked reference."""
import torch
import torch.nn.functional as F

from edenlive.models.adapters.wan22_diffusers.dynamic_prompt import (
    _sdpa_block_attention,
)

torch.manual_seed(0)
B, n_blk, L, H, D = 2, 5, 7, 3, 8  # 2 samples, 5 segment-blocks total, text_length=7
T_q = B * n_blk * (L // 1)  # 25 query tokens each in our toy layout
# toy: q_flat [B*T_b, H, D], packed per sample
nq_per = n_blk
q_flat = torch.randn(B * nq_per, H, D, dtype=torch.float32)
packed = torch.randn(B * nq_per * L, H, D, dtype=torch.float32)
cu = [0]
for _ in range(B):
    for _ in range(nq_per):
        cu.append(cu[-1] + L)
cu_q = torch.tensor(cu, dtype=torch.int32)

out = _sdpa_block_attention(q_flat, packed, packed, cu_q, L)

# naive reference: for each flat block j: q tokens cu[j]:cu[j+1] attend K [j*L,(j+1)*L)
ref = torch.empty_like(q_flat)
for j in range(B * nq_per):
    a, b = cu[j], cu[j + 1]
    k0, k1 = j * L, (j + 1) * L
    qj = q_flat[a:b]
    kj = packed[k0:k1]
    vj = packed[k0:k1]
    ref[a:b] = F.scaled_dot_product_attention(
        qj.transpose(0, 1), kj.transpose(0, 1), vj.transpose(0, 1)
    ).transpose(0, 1)

assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()
print("OK  _sdpa_block_attention matches naive reference, max err:", (out - ref).abs().max().item())
