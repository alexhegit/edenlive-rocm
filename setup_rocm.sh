#!/usr/bin/env bash
# EdenLive on AMD ROCm (MI300X, gfx90a) — environment bootstrap.
#
# Scope: OFFLINE few-step (chunk-autoregressive) inference with the 5B model
# (wan22_5b_720p_few_step.yaml). That path is pure PyTorch + diffusers native
# attention ("triton" backend in configs = dispatch_attention_fn(backend=None)
# = SDPA/ROCm flash-attention). The NVIDIA-only parts (FA4 / cutlass, SM90,
# the cu130 torch install, the LightX2V stack) are all in the *bidirectional*
# path and the RTX-5090 real-time service, which this setup does NOT use.
#
# Verified 2026-11-06 on: MI300X, ROCm 7.2.4, torch 2.10.0+rocm7.2.4,
# container image: rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
set -euo pipefail

REPO="${1:-$PWD/EdenLive}"
IMAGE="${2:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
CONTAINER="${3:-edenlive-rocm}"

# 1) Container with the GPU passthrough flags ROCm needs.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "$REPO":/mnt/doscratch/EdenLive -w /mnt/doscratch/EdenLive \
  "$IMAGE" sleep infinity

run() { docker exec "$CONTAINER" bash -lc "$*"; }

# 2) Install the package WITHOUT its torch pin (keep the ROCm build), add the
#    rest of the dependencies, plus the two video-decode libraries the data
#    loader imports at runtime. NOTE: the repo's requirements/wan22-fa4*.txt
#    (flash-attn-4 / nvidia-cutlass-dsl) are CUDA-only and must NOT be installed.
run "pip install --no-deps -e ."
run "pip install -U accelerate diffusers ftfy huggingface-hub imageio imageio-ffmpeg numpy pillow PyYAML safetensors transformers pandas pyarrow av==17.1.0 decord==0.6.0"

# 3) 5B weights only (~40 GB). Skip dit-14b*/vae-14b*/clip (14B) and flux-c2f
#    (FLUX non-commercial, workbench-only).
#    GOTCHA: one `hf download ... --include a b c` silently drops some patterns
#    ("Ignoring --include since filenames have been explicitly set"), so the
#    ~20 GB dit-5b weights can be missing. One -I/--include per invocation.
for f in 'dit-5b' 'dit-5b/*' 'vae-5b/*' 'decoder-5b/*' 'scheduler-5b/*' \
         'umt5/*' 'vision-encoder/*'; do
  run "pip install -q huggingface_hub && hf download VAST-AI/EdenLive --local-dir ./models/EdenLive --include '$f'"
done
# Verify the DiT actually landed (the failure mode above is a 16 GB download
# that looks complete but misses diffusion_pytorch_model.safetensors).
run "test -s ./models/EdenLive/dit-5b/diffusion_pytorch_model.safetensors"

# 3b) Optional: 14B @ 480p few-step (~65 GB extra: dit-14b 62 GB + vae-14b,
#     CLIP ViT-H/14 image encoder for Wan2.1-I2V + scheduler-14b).
#     Verified working on a single MI300X (peak VRAM ~115 GB).
for f in 'dit-14b/*' 'vae-14b/*' 'clip/*' 'scheduler-14b/*'; do
  run "pip install -q huggingface_hub && hf download VAST-AI/EdenLive --local-dir ./models/EdenLive --include '$f'"
done
run "test -s ./models/EdenLive/dit-14b/diffusion_pytorch_model.safetensors"

echo "==================================================================="
echo "Environment ready. Run the ROCm-validated offline path:"
echo
echo "docker exec edenlive-rocm bash -lc \"cd \$REPO && python -u -m edenlive.scripts.infer \\"
echo "  --config edenlive/configs/wan22_5b_720p_few_step.yaml \\"
echo "  data.dataset.params.path=examples/samples/manifest.csv \\"
echo "  data.dataset.params.root=examples/samples\""
echo
echo "Known boundaries (not runnable here):"
echo "  - *bidir* configs (5B 720p AND 14B 480p) require flash-attn-4 (SM90/Hopper);"
echo "    CUDA-only. The 14B *few-step* config is verified working (see above)."
echo "  - Real-time WebRTC service at 24 FPS: built around CUDA latency tuning; expect misses on ROCm."
echo "==================================================================="
