# EdenLive on AMD ROCm (MI300X / gfx90a)

Ported and **verified running** 2026-11-05 (few-step) and 2026-11-06
(bidir, after the `flash4 → SDPA` fallback port) on 5B + 14B models,
offline inference.

## What works on MI300X

### 5B — `wan22_5b_720p_bidir.yaml` (Wan2.2-TI2V-5B @ 720p, **full-quality bidirectional**) ✅ verified 2026-11-06

1280×704, **121 frames, 48 steps, `flow_unipc`, bidirectional denoising** —
the "quality" path that was CUDA-gated by `flash4`. Now runs on MI300X
via the SDPA fallback (see `## Code changes for the bidir port` below):

- `teacher_forcing_attention_backend: flash4` → auto-falls back to
  `F.scaled_dot_product_attention` (numerically equivalent unmasked
  self-attention; verified against a naive reference, max err 0.0) ✅
- `dynamic_prompt: enabled` (per-frame-block text conditioning) →
  FA4 varlen kernel replaced by a per-block SDPA loop
  (`_sdpa_block_attention`) ✅
- 2× sample generate + VAE decode + libx264 encode ~ **3-4 min per
  121-frame clip end-to-end** (~0.6 FPS; the 48-step bidir schedule is
  ~12× more model-fwd than 4-step few-step, so this is expected) ✅
- `PYTHON_EXIT=0`, content verified non-flat (per-frame std 0.23-0.28,
  channel stds 0.20-0.29, inter-frame diff 0.05-0.26) ✅
- Requires the extra weight dir `dit-5b-bidir/` (~19 GB) on top of the
  few-step set.

### 14B — `wan21_14b_480p_bidir.yaml` (Wan2.1-I2V-14B @ 480p, bidirectional) ✅ verified 2026-11-06

832×480, 81 frames, 40 steps, `flow_unipc` — the 14B "quality" path.
Uses `teacher_forcing_attention_backend: triton` (native SDPA, no `flash4`
dependency) and `dynamic_prompt.enabled: false`, so it needs **no fallback**
— plain single-card run, same overrides as the 14B few-step config. Both
samples completed `PYTHON_EXIT=0` with real content (frame std 0.27,
per-channel 0.23–0.29, inter-frame diffs 0.04–0.24), ~7.5 min end-to-end
per 81-frame clip. Requires `dit-14b-bidir/` (~64 GB).

### 5B — `wan22_5b_720p_few_step.yaml` (Wan2.2-TI2V-5B @ 720p)

1280×704, 717 frames, 4-step chunk-AR. Full pipeline on a single MI300X
(192 GB VRAM, ROCm 7.2.4, torch
2.10.0+rocm7.2.4, container
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`):

- Data loading (av + decord, recording_manifest dataset) ✅
- 5B DiT + Wan VAE + UMT5 + proxy-DINOv3 encoder load ✅
- `teacher_forcing_attention_backend: triton` path ✅
  (this is `dispatch_attention_fn(backend=None)` = native/SDPA flash
  attention; the FA4/CuTe path is *not* touched)
- 2× sample generate + VAE decode + libx264 encode ✅
  (measured ~115-150 s per 717-frame clip end-to-end incl. data prep + save)
- Outputs saved to `inference/wan22_5b_720p_few_step/save/inference/base/`
  (`<id>.mp4`, `<id>_proxy.mp4`, `<id>_image.png`, `<id>_prompt.txt`) ✅

### 14B — `wan21_14b_480p_few_step.yaml` (Wan2.1-I2V-14B @ 480p)  ✅ verified 2026-11-05

832×720, 477 frames, 4-step chunk-AR, **single MI300X card** (official config
asks for 2-GPU sequence parallel + FSDP2; both disabled via CLI overrides):

```
inference.max_batches=1 runtime.fsdp2.enabled=false runtime.sequence_parallel.size=1
DIFFUSERS_ATTN_BACKEND=_native_flash   # required: must be set BEFORE python starts
```

- Extra components vs 5B: **CLIP ViT-H/14 image encoder/processor** (Wan2.1
  I2V needs it), `scheduler-14b` (scheduler_config.json, a few KB — easy to
  forget in the downloads, `FileNotFoundError` at load if missing)
- 62 GB DiT bf16 + 2.7 GB VAE + 11 GB UMT5 + CLIP → peak VRAM **~115 GB**,
  fits one 192 GB card with plenty of headroom for KV cache + activations ✅
- generate 477 frames @480p, ~6 min end-to-end, `PYTHON_EXIT=0`, content
  verified (per-channel std 44, frame-to-frame diff 97) ✅
- One benign warning: `torch.rms_norm` dtype mismatch (fp32 input vs bf16
  weight) — numerical path, not a failure.

Measured throughput ≈ 717 frames / 115 s ≈ **6.2 FPS** end-to-end
(data + encode included). DiT-only would be higher. This is far below the
29.8 FPS on H100 or 24.7 FPS on RTX 5090 that the paper reports, because:

1. Those numbers are **model-side inference throughput on H100/5090**,
   measured with `LightX2V` CUDA-Graph + quant + fused kernels.
   We are on plain ROCm `SDPA`, no CUDA graphs, no quant.
2. MI300X is compute-bound on bf16 GEMM; the DiT is a wide 5B transformer,
   so per-token cost is comparable to H100, but `LightX2V` speedups (which
   account for most of the real-time margin) are NVIDIA-only.
3. Real-time 24 FPS on this port **will not be met** without a LightX2V
   equivalent on ROCm (a separate project).

## What does NOT work on MI300X (and is not attempted)

| Component | Why |
|---|---|
| `realtime/` WebRTC service (24 FPS target) | `flashdreams`/`apps/cam2v` + LightX2V accelerated decode stack — no ROCm equivalent. The 5× real-time margin in the paper comes from CUDA graphs + fused kernels + quant; that whole stack is NVIDIA-only. |
| `reactor/` (B200) | Runs on 1× B200 by design. |
| `flux-c2f/` first-frame renderer | FLUX.2 non-commercial, workbench-only. Irrelevant for offline rendering. |

Both `*bidir*` configs **now work** via the fallbacks described above;
what remains gated is the *real-time speed*, not the functionality.

## Files added

- `rocm/setup_rocm.sh` — one-shot container + deps + 5B weights bootstrap.
- `rocm/probe1.py` — minimal end-to-end probe (single sample, saves under
  `outputs/probe/`). Use it to bisect regressions.
- `rocm/dscheck.py` — dataset + dataloader smoke test without the model (no
  GPU needed once weights are on disk; validates `recording.mp4` decoding +
  aspect-ratio check + chunk_prompts shape).
- `rocm/README.md` — this file.

## Setup (short form)

```bash
# 1) Container with GPU passthrough.
docker run -d --name edenlive-rocm \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $REPO:/mnt/doscratch/EdenLive -w /mnt/doscratch/EdenLive \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  sleep infinity

# 2) Install without the cu130 torch pin (keep ROCm build).
docker exec edenlive-rocm bash -lc "cd /mnt/doscratch/EdenLive && \
  pip install --no-deps -e . && \
  pip install -U accelerate diffusers ftfy huggingface-hub imageio imageio-ffmpeg \
    numpy pillow PyYAML safetensors transformers pandas pyarrow av==17.1.0 decord==0.6.0"

# 3) 5B weights (~40 GB), one --include per call (multi-pattern is a no-op).
for f in dit-5b dit-5b/* vae-5b/* decoder-5b/* scheduler-5b/* umt5/* vision-encoder/*; do
  docker exec edenlive-rocm bash -lc "cd /mnt/doscratch/EdenLive && \
    pip install -q huggingface_hub && \
    hf download VAST-AI/EdenLive --local-dir ./models/EdenLive --include '$f'"
done
```

## Run (the validated path)

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 edenlive-rocm bash -lc "
  cd /mnt/doscratch/EdenLive && \
  python -X faulthandler -u -m edenlive.scripts.infer \
    --config edenlive/configs/wan22_5b_720p_few_step.yaml \
    data.dataset.params.path=examples/samples/manifest.csv \
    data.dataset.params.root=examples/samples"
```

Outputs → `inference/wan22_5b_720p_few_step/save/inference/base/<id>.mp4`.

**14B @ 480p (single card, verified):**

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 -e DIFFUSERS_ATTN_BACKEND=_native_flash edenlive-rocm bash -lc "
  cd /mnt/doscratch/EdenLive && \
  python -X faulthandler -u -m edenlive.scripts.infer \
    --config edenlive/configs/wan21_14b_480p_few_step.yaml \
    data.dataset.params.path=examples/samples/manifest.csv \
    data.dataset.params.root=examples/samples \
    inference.max_batches=1 \
    runtime.fsdp2.enabled=false \
    runtime.sequence_parallel.size=1"
```

Note `DIFFUSERS_ATTN_BACKEND=_native_flash` is mandatory for this config
(`runtime.attention_backend.backend: _native_flash`) and must be set in the
process environment *before* python starts — the plugin refuses a post-start
switch. Outputs → `inference/wan21_14b_480p_few_step/save/inference/base/<id>.mp4`.

## Pitfalls we hit (keep these when extending)

1. **`hf download --include a b c`** is interpreted as "explicit filenames are
   set" and *ignores* `--include` — one pattern per call, then verify
   `diffusion_pytorch_model.safetensors` exists (16 GB "complete" download
   that's missing the 20 GB DiT is indistinguishable by eye).
2. **`recording_manifest` dataset** reads `root / row["proxy_video"]` and
   expects `root/row_dir/recording.mp4` + `root/row_dir/prompts.json` +
   `cover.png` / `first-frame.png` layout. The shipped manifest uses
   `path=<id>/recording.mp4` style — pass `root=examples/samples` (dir that
   *contains* the `<id>/` subdirs), not `root=examples/samples/<id>`.
   Aspect ratio must be within 20% of 1280:704.
3. **The `triton` backend string in the config does NOT mean Triton
   kernels** — it's `dispatch_attention_fn(backend=None)` → Diffusers
   native SDPA. It *looks* like a Triton path, but on ROCm it just picks
   the default SDPA flash kernel. Do not add a Triton kernel under the
   impression this is the fast path.
4. **`max_sequence_length: 512`** in the wan22 config caps text; long
   multi-chunk prompts (60 chunk_prompts) work because token length is
   per-chunk, but if you raise `num_frames` or `chunk_size`, re-check the
   UMT5 context window before assuming it's fine.
5. **Sequence parallelism** (`runtime.sequence_parallel.enabled: true` with
   `size: 1`) is a no-op on one GPU; it will engage on `torchrun
   --nproc_per_node=2`. `runtime.fsdp2.enabled: false` — leave it off
   unless you actually hit the 192 GB ceiling.
6. **Exit-code masking.** When piping `infer ... | tail`, the shell reports
   *tail's* exit code — a failed `python` step is invisible in `$?`, and
   without `python -u` an early crash's buffered traceback can also be lost.
   Log to a file and capture `${PIPESTATUS[0]}`.
7. **The `flux-c2f` component in the 210 GB full download is
   FLUX non-commercial.** We skipped it; the offline video pipeline does
   not load it (confirmed by `patch_wan_transformer_forward` + adapter code
   paths). Do not ship a "run the whole workbench" story under Apache 2.0.
8. **`DIFFUSERS_ATTN_BACKEND` must be set before python starts.** The
   `AttentionBackendPlugin` refuses any mismatch between the config's
   `runtime.attention_backend.backend` (14B few-step says `_native_flash`)
   and `os.environ.setdefault` value. The 5B config has
   `attention_backend.enabled: false` so the env var is NOT required — do
   not copy the 14B launch form onto the 5B run (it still works there, but
   the env-var name `_native_flash` is a diffusers *internal* alias for the
   SDPA flash kernel and is not documented by AMD as a stable ROCm setting;
   treat it as "native SDPA flash" and re-check after a diffusers version
   bump).
9. **`scheduler-14b/scheduler_config.json` is a few-KB file** — it is easy to
   forget when doing per-directory `hf download --include`. The pipeline
   build dies immediately with a `FileNotFoundError` at the components
   stage. Verify every components list in the model config against the
   filesystem before running a big job (one-liner in the container:
   `ls models/EdenLive/<path>` for each line of the config's
   `components:` block).

## Code changes for the bidir port (P0)

Two files in `edenlive/models/adapters/wan22_diffusers/`:

1. `teacher_forcing_attention.py`
   - `validate_flash4_environment(device, dtype)` now **returns** `True/False`
     instead of raising: ROCm (`torch.version.hip` is not None) or any
     non-CUDA platform returns `False` (fallback intent); CUDA+Hopper keeps
     the original strict gates (SM90, BF16, FA4 version check) so validated
     H100 behaviour is unchanged.
   - New `_detect_fa4_kernel()` — lazy, cached check for whether the CuTe
     FA4 wheel is importable on this platform.
   - `flash4_attention(q,k,v)` dispatches: FA4 kernel when
     `_detect_fa4_kernel()` is true, else a numerically-equivalent
     `F.scaled_dot_product_attention` path (unmasked self-attention). The
     caller (`wan_ops.py:166-191`) is unchanged.

2. `dynamic_prompt.py`
   - `cross_attention(...)` dispatches the same way: FA4 varlen kernel when
     available, else `_sdpa_block_attention` — a tight per-frame-block SDPA
     loop that reproduces the `dynamic_prompt` contract (each query block
     attends to its corresponding text segment K/V window `text_length`
     wide). The `cu_q` / `lengths` / `text_length` contract the FA4 kernel
     was using is preserved; a naive reference implementation reproduces it
     to `max err 0.0` on a toy 2×5×7 layout (see `rocm/test_sdpa_fallback.py`).
   - `validate_environment(device, dtype)` no longer raises on ROCm; it
     logs the fallback intent and returns, so the 5B bidir config loads
     cleanly.

Test / reproduction:

- `rocm/test_sdpa_fallback.py` — pure-attention SDPA fallback numerical
  check vs. a masked reference (passes).
- `rocm/check_bidir_video.py` — inspect a bidir output clip (frame/std/
  channel/inter-frame diffs) to confirm the model produced real content,
  not flat frames.

```bash
# 5B bidir (720p, 121 frames, 48 steps) — ~3-4 min per sample on MI300X
docker exec -e HIP_VISIBLE_DEVICES=0 edenlive-rocm bash -lc "
  cd /mnt/doscratch/EdenLive && \
  python -X faulthandler -u -m edenlive.scripts.infer \
    --config edenlive/configs/wan22_5b_720p_bidir.yaml \
    data.dataset.params.path=examples/samples/manifest.csv \
    data.dataset.params.root=examples/samples"

# 14B bidir (480p, 81 frames, 40 steps) — ~7.5 min per sample on MI300X
docker exec -e HIP_VISIBLE_DEVICES=0 -e DIFFUSERS_ATTN_BACKEND=_native_flash edenlive-rocm bash -lc "
  cd /mnt/doscratch/EdenLive && \
  python -X faulthandler -u -m edenlive.scripts.infer \
    --config edenlive/configs/wan21_14b_480p_bidir.yaml \
    data.dataset.params.path=examples/samples/manifest.csv \
    data.dataset.params.root=examples/samples \
    inference.max_batches=1 \
    runtime.fsdp2.enabled=false \
    runtime.sequence_parallel.size=1"
```

Both `*bidir*` runs need the extra per-model bidir weight dirs (`dit-5b-bidir/`
~19 GB, `dit-14b-bidir/` ~64 GB) which are separate from the few-step dirs.
`hf download VAST-AI/EdenLive --include 'dit-<size>-bidir/*' --local-dir <repo>/models/EdenLive`
(one `--include` per call — see pitfalls §1 above).

## P1 speedup trial: torch.compile (inductor) on MI300X — measured, honest

Inductor **does** compile and run the DiT here (verified with a standalone
smoke test, `rocm/p1_inductor_smoke.py`, before committing to full runs —
that de-risked the path). Results for 5B @ 720p:

| Path | baseline (eager) | `runtime.torch_compile.enabled=true` | speedup |
|---|---|---|---|
| **bidir** (121 f, 48 steps, no cache, full-SDPA) | ~216 s / sample (sample 2) | ~170 s / sample (sample 2) | **~1.27×** |
| **few-step** (717 f, 4 steps, cache-chunk-AR) | ~115–150 s | ~124 s | **~1.0× (flat)** |

Why the split: the bidir pass is unmasked full-attention with no per-step
KV-cache shape churn, so inductor can fuse it; the few-step chunk-AR pass
has dynamic KV-cache shapes that fragment the compile graph, so it gains
little. A single isolated attention block measures ~1.46× under inductor —
that's the pure-attention ceiling, and end-to-end is diluted by VAE decode
+ libx264 (uncompiled, fixed cost) and the cache path.

Turn it on (no code change; the repo already ships the `TorchCompilePlugin`):
```bash
... --config edenlive/configs/wan22_5b_720p_bidir.yaml \
    runtime.torch_compile.enabled=true inference.output_dir=<dir-with-compile>
```
First sample pays the compile wall (~30–60 s once per process; cached to
`~/.cache/torchinductor_root` after that). Both paths content-verified after
compile (per-frame std ~0.26–0.27, non-flat).

The ~5× of the H100 paper still needs the LightX2V stack (CUDA-graphs +
fused + quant); inductor is the available ~1.3×.

## P2 trial: W8A8/FP8 quantization via torchao — measured, mixed

`pip install torchao` (0.18.0) on the `pytorch/rocm:rocm7.2.4` container
**skips its cpp extensions** because they require `torch >= 2.11` (we have
2.10.0). So torchao runs on the pure-Python fallback path.

**Isolated GEMM microbench** (`rocm/p2_ao_bench.py`, 5B DiT MLP shape
`4096→14336`, H100-style K=2048):

| recipe | ms/GEMM | vs bf16 (0.136 ms) | cosine |
|---|---|---|---|
| bf16 (anchor) | 0.136 | 1.00× | — |
| Float8DynamicActivationFloat8Weight (eager) | 0.110 | **1.23×** | 0.99929 |
| Float8DynamicActivationFloat8Weight (+inductor) | 0.083 | **1.64×** | 0.99932 |
| Int8WeightOnly (eager) | 0.175 | 0.78× | 0.99983 |
| Int8WeightOnly (+inductor) | 0.162 | 0.84× | 0.99979 |
| Int8DynamicActivationInt8Weight (eager) | 0.839 | 0.16× | 0.99884 |
| Int8DynamicActivationInt8Weight (+inductor) | 0.324 | 0.42× | 0.99885 |

Takeaway: **native FP8 (CDNA3) is the working recipe**; torchao's int8 fast
path is not connected in this build (cpp ext skipped) and is either flat or a
regression.

**End-to-end, the win does not materialize.** 5B bidir @ 720p, 2 samples:

| Config | sample 2 |
|---|---|
| bf16 eager (P0) | ~216 s |
| bf16 + `torch_compile` (P1) | ~170 s |
| **fp8_w8a8 + `torch_compile` (P2)** | **~219 s** |

Reason: the per-layer dynamic FP8 activation quant/dequant work happens in
Python under the fallback path, and its per-layer overhead is larger than the
GEMM saving. The microbench is honest (it only measures the GEMM), but
end-to-end the VAE decode + libx264 + Python-side quant wrappers dominate.
This is the expected outcome for torchao-on-<torch-2.11; we keep the plugin
so it can be re-enabled the moment a `torch >= 2.11` ROCm container lands.

**What we still ship** (no behavior change unless explicitly turned on):
- `edenlive/runtime/quantization.py` — `AOQuantizePlugin`, a
  `runtime.ao_quantize.*` surface: `enabled`, `recipe` (`fp8_w8a8` |
  `int8_w8a8` | `int8_weight_only` | `int4_weight_only`),
  `transformer_only` (default true: only the DiT, not VAE/text encoder).
- Registration in `edenlive/runtime/__init__.py` and added to the
  `runtimes` tuple of the Wan22DiffusersAdapter.
- Ordering: `ao_quantize` prepares **before** `torch_compile`, so the
  inductor graph already contains the 8-bit GEMM ops (same order as
  training; see `edenlive/inference/offline.py`).
- Enable with:
  `runtime.ao_quantize.enabled=true runtime.ao_quantize.recipe=fp8_w8a8`
  (optional alongside `runtime.torch_compile.enabled=true`).

Content check after quantization is identical in spirit to P1: 121 f @
1280×704, per-frame std ~0.26–0.27, non-flat, motion present in both
samples (2 exit 0).

**Practical recommendation today:** do **not** enable `ao_quantize` for
production. Use `torch_compile.enabled=true` (P1's ~1.27× on bidir). Re-run
with a `torch>=2.11` ROCm image to re-test, or look at a vendor-native
`rccl`/`qut` FP8 kernel that skips torchao's Python fallback entirely.

## What would be needed to go faster on MI300X (future, out of scope)

- LightX2V-equivalent ROCm kernels + CUDA-Graph-equivalent (rocprof-comgr
  graphs) for the DiT — this is the ~5× that the paper gets on H100.
- A ROCm `pytorch/rocm:torch-rocm-2.11+` style image so that
  `pip install torchao` compiles its fast cpp kernels — that is the path
  where FP8 W8A8 (the 1.64× microbench result here) can actually pay off
  end-to-end, instead of running on the pure-Python fallback.
- Sequence-parallel across 2 MI300X for the 14B bidir config — the config
  file is already correct, but the FA4 path in bidir is the blocker; would
  need a Triton/ROCm flash-attention replacement for the bidir kernel.
