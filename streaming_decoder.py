"""P3 M1: chunk-by-chunk streaming decode of EdenLive few-step chunk-AR latents.

Hook into ``manager.generate(..., on_chunk_ready=...)`` (added in M1) and,
as each chunk commits its latents, decode just its slice — with a causal
look-back prefix of ``lookback_latent_frames`` (default 2) — to obtain the
corresponding pixel frames and report the first-frame latency.

The Wan decoder is causal in time (verified by ``rocm/m1_vae_chunk_probe.py``
on the real 5B VAE on MI300X):
  * a latent segment ``z[:, :, a:b]`` (b>a) decodes to ``4(b-a)-3`` frames,
  * re-decoding with an ``L``-frame look-back prefix and dropping the
    redundant prefix output is numerically close to decoding the whole
    sequence at once (probe: 3.4-3.8% rel diff on bf16, max 0.73 on a 96x96
    25-frame clip — visually indistinguishable in preview use).

The point of this hook is **first-byte latency for interactive streaming**.
We don't re-write ``offline.py``'s "decode the whole latent at the end"
contract — the streaming hook is purely additive and, if it is not supplied,
the offline pipeline behaves exactly as before.

API
----
::

    from edenlive.inference.streaming import ChunkedVAEDecoder
    decoder = ChunkedVAEDecoder(vae, latents, chunk_size, lookback_frames=2)
    manager.generate(..., on_chunk_ready=lambda info: decoder.on_chunk(info))
    frames = decoder.collected_frames      # torch.Tensor [1, C, T, H, W] float in [-1,1]
    first_frame_latency_ms = decoder.first_frame_latency_ms
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from collections import deque


@dataclass
class ChunkStreamStats:
    """Per-chunk and per-rollout latency / pixel stats for M1 verification."""

    chunk_latent_span: list[list[int]] = field(default_factory=list)
    chunk_decode_ms: list[float] = field(default_factory=list)
    chunk_frames: list[int] = field(default_factory=list)
    first_frame_emitted_at_seconds: float | None = None
    t_first_chunk_ready: float | None = None
    t_first_frame_emitted: float | None = None

    @property
    def first_frame_latency_ms(self) -> float | None:
        if self.t_first_chunk_ready is None or self.t_first_frame_emitted is None:
            return None
        return (self.t_first_frame_emitted - self.t_first_chunk_ready) * 1e3


class ChunkedVAEDecoder:
    """Decode committed latent chunks with a causal look-back prefix.

    ``lookback_latency`` is the number of *latents* re-decoded (and then
    dropped) before the target slice.  Two latents of lookback was what the
    M1 smoke probe (``rocm/m1_vae_chunk_probe.py``) validated as giving
    near-exact numbers while being the minimum that covers the causal
    first-frame rule of the Wan decoder.
    """

    def __init__(
        self,
        vae: Any,
        latents: torch.Tensor,
        chunk_size: int,
        *,
        lookback_latency: int = 2,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latents_mean: torch.Tensor | Any = None,
        latents_std: torch.Tensor | Any = None,
    ) -> None:
        if vae is None:
            raise ValueError("vae is required")
        if latents.dim() != 5:
            raise ValueError(f"latents must be [B,C,T,H,W], got {tuple(latents.shape)}")
        self.vae = vae
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.lookback_latency = int(lookback_latency)
        if self.lookback_latency < 0:
            raise ValueError("lookback_latency must be >= 0")
        self.b = int(latents.shape[0])
        self.c = int(latents.shape[1])
        self.total_latent_frames = int(latents.shape[2])
        self.latent_height = int(latents.shape[3])
        self.latent_width = int(latents.shape[4])
        self.device = torch.device(device) if device is not None else latents.device
        self.dtype = dtype if dtype is not None else latents.dtype
        if latents_mean is None:
            mean = list(getattr(vae.config, "latents_mean", []))
            self.latents_mean = torch.tensor(mean, dtype=self.dtype, device=self.device).view(1, self.c, 1, 1, 1)
        else:
            self.latents_mean = latents_mean.to(device=self.device, dtype=self.dtype)
        if latents_std is None:
            std = list(getattr(vae.config, "latents_std", []))
            self.latents_std = torch.tensor(std, dtype=self.dtype, device=self.device).view(1, self.c, 1, 1, 1)
        else:
            self.latents_std = latents_std.to(device=self.device, dtype=self.dtype)
        self.commit_latents = latents
        self._chunks: deque[torch.Tensor] = deque()
        self.stats = ChunkStreamStats()
        self._next_chunk_start = 0

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def on_chunk(self, info: Any) -> None:
        """Callback for ``manager.generate(..., on_chunk_ready=...)``."""
        import time as _time

        info_start = int(getattr(info, "chunk_start", self._next_chunk_start))
        info_frames = int(getattr(info, "chunk_frames", self.chunk_size))
        info_total = int(getattr(info, "total_latent_frames", self.total_latent_frames))
        _t0 = _time.perf_counter()
        torch.cuda.synchronize()

        prefix_start = max(0, info_start - self.lookback_latency)
        end = info_start + info_frames
        seg = self.commit_latents[:, :, prefix_start:end]
        if self.device.type == "cuda":
            seg = seg.to(device=self.device, dtype=self.dtype, non_blocking=True)
        z = seg / self.latents_std + self.latents_mean
        with torch.no_grad():
            out = self.vae.decode(z, return_dict=True).sample

        # drop the re-decoded look-back prefix (keep the target chunk's frames)
        if info_start == prefix_start:
            keep_from = 0
        else:
            # causal mapping: n latents -> 4n-3 frames, first latent -> 1 frame
            lookback_frames = 4 * (info_start - prefix_start) - 3
            keep_from = max(0, lookback_frames)
        out = out[:, :, keep_from:]
        out = out.detach().float().cpu()
        self._chunks.append(out)
        torch.cuda.synchronize()
        _t1 = _time.perf_counter()

        # bookkeeping
        self.stats.chunk_latent_span.append([prefix_start, end])
        self.stats.chunk_decode_ms.append((_t1 - _t0) * 1e3)
        self.stats.chunk_frames.append(int(out.shape[2]))
        if self.stats.t_first_chunk_ready is None:
            self.stats.t_first_chunk_ready = _t0
        if self.stats.t_first_frame_emitted is None and out.numel() > 0:
            self.stats.t_first_frame_emitted = _t1

        if info_start + info_frames >= info_total:
            self._mark_finished()

    def _mark_finished(self) -> None:
        pass

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def collected_frames(self) -> torch.Tensor:
        if not self._chunks:
            raise RuntimeError("no chunks have been decoded yet")
        return torch.cat(list(self._chunks), dim=2)

    def __repr__(self) -> str:
        s = self.stats
        ff = f"{s.first_frame_latency_ms:.1f} ms" if s.first_frame_latency_ms is not None else "n/a"
        return (
            f"ChunkedVAEDecoder(chunks={len(s.chunk_latent_span)}, "
            f"total_frames={sum(s.chunk_frames) if s.chunk_frames else 0}, "
            f"first_frame_latency={ff})"
        )


# ------------------------------------------------------------------------
# convenience helpers for ad-hoc streaming demos / WebSocket servers (M2)
# ------------------------------------------------------------------------
def make_on_chunk_ready(decode_callback: Callable[[torch.Tensor], None], decoder: ChunkedVAEDecoder):
    """Compose an ``on_chunk_ready`` callable that decodes + hands the slice
    to a downstream user callback.  The user callback receives the *fresh*
    pixel slice (not the cumulative tensor).

    Example (JPEG streaming, M2)::

        dec = ChunkedVAEDecoder(pipe.vae, latents, chunk_size=3)
        import av, io
        def send_jpeg(chunk_frames: torch.Tensor):  # [1, C, t, H, W]
            ...  # encode to JPEG, push over WebSocket
        handle = make_on_chunk_ready(send_jpeg, dec)
        manager.generate(..., on_chunk_ready=handle)
    """
    def _on_chunk(info: Any) -> None:
        decoder.on_chunk(info)
        # pass only the *newly ready* frames downstream
        frames = decoder.collected_frames
        last_chunk = decoder.stats.chunk_frames[-1]
        if last_chunk > 0:
            chunk_frames = frames[:, :, -last_chunk:]
            decode_callback(chunk_frames)
    return _on_chunk_ready
