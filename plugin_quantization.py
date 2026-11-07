"""AOQuantizePlugin — W8A8 weight/activation quantization for the DiT.

Targets only the ``transformer`` (DiT) in the pipeline; VAE and text encoder
are left untouched (their precision is more sensitive to 8-bit loss and they
are a small fraction of end-to-end time).

Supported recipes (torchao 0.18.x, ROCm 7.2.4 build):
- ``fp8_w8a8``   — Float8DynamicActivationFloat8WeightConfig
  (MI300X CDNA3 native FP8; measured ~1.23× eager / ~1.64× with inductor
  vs 5B DiT GEMM shape, cosine 0.9993)
- ``int8_weight_only`` — Int8WeightOnlyConfig (measured ~0.78× — usually
  slower than bf16 because the dequant path dominates in this build)
- ``int8_w8a8``  — Int8DynamicActivationInt8WeightConfig (measured ~0.16×
  eager / 0.42× w/ inductor; torchao 0.18 int8 kernel path is not well
  connected on ROCm 7.2.4. Kept supported for future kernel fixes, but
  documented as a *no-improvement* path today.)

``enabled: false`` (default) → plugin is a no-op, so it does not affect any
other config's behaviour. Turning it on is an explicit CLI override:

    runtime.ao_quantize.enabled=true
    runtime.ao_quantize.recipe=fp8_w8a8          # default
    runtime.ao_quantize.transformer_only=true    # only quantize the DiT

Combines well with ``runtime.torch_compile.enabled=true`` in the order:
    ao_quantize (prepare: quantize Linears in place)  →  torch_compile (prepare:
    compile block.forward).  The compiled graph then sees the quantized
    modules; the benchmark in P2 shows ~1.64× for FP8_w8a8 when both are on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from edenlive.core.types import RuntimeContext
from edenlive.runtime.base import RuntimePlugin
from edenlive.utils import get_logger

if TYPE_CHECKING:
    from edenlive.models.base import ModelAdapter

logger = get_logger(__name__)


class AOQuantizePlugin(RuntimePlugin):
    """Quantize DiT Linear modules via torchao (FP8 W8A8 by default on CDNA3)."""

    name = "ao_quantize"

    @dataclass
    class Config:
        enabled: bool = False
        # One of: fp8_w8a8 (default, fastest) | int8_w8a8 | int8_weight_only | int4_weight_only
        recipe: str = "fp8_w8a8"
        # Restrict quantization to ``pipeline.transformer`` (the DiT).
        # True (default): VAE + text_encoder stay bf16 (their precision is
        # more sensitive and they are small anyway).
        transformer_only: bool = True
        # Extra args forwarded to the recipe Config (e.g. group_size, block_size).
        extra: dict[str, Any] = field(default_factory=dict)

    def _resolve_recipe(self, recipe: str):
        """Return a torchao Config *instance* for the given recipe name."""
        import torchao.quantization.quant_api as qa

        cfg_map = {
            "fp8_w8a8": qa.Float8DynamicActivationFloat8WeightConfig,
            "int8_w8a8": qa.Int8DynamicActivationInt8WeightConfig,
            "int8_weight_only": qa.Int8WeightOnlyConfig,
            "int4_weight_only": qa.Int4WeightOnlyConfig,
        }
        if recipe not in cfg_map:
            raise ValueError(
                f"runtime.ao_quantize.recipe must be one of "
                f"{sorted(cfg_map)}, got {recipe!r}"
            )
        return cfg_map[recipe]

    def prepare_model(self, model: ModelAdapter, context: RuntimeContext) -> None:
        if not self.enabled:
            return
        import torch
        from torchao.quantization import quantize_

        pipe = model.components.get("pipeline")
        if pipe is None:
            raise RuntimeError("ao_quantize: pipeline not loaded yet")

        targets: list[torch.nn.Module] = []
        if self.config.transformer_only:
            for key in ("transformer",):
                m = getattr(pipe, key, None)
                if m is not None:
                    targets.append(m)
        if not targets:
            # Fall back to the whole pipeline (VAE + text_encoder too).
            # This is NOT recommended — see Config.transformer_only.
            targets.append(pipe)
            logger.info_rank0(
                "ao_quantize: no transformer found on pipeline; quantizing the whole pipeline (VAE/text_encoder included) — accuracy drop is likely to be larger."
            )

        cfg_cls = self._resolve_recipe(self.config.recipe)
        cfg = cfg_cls(**self.config.extra)
        for target in targets:
            # torchao replaces the linear modules in place. Surface the real
            # error rather than silently skipping: int4 needs the ``mslk``
            # wheel, and some int8 fast paths need torch.compile + torch >= 2.11
            # (torchao cpp extensions are skipped on older torch, leaving a
            # slower pure-Python path).
            quantize_(
                target, cfg,
                device="cuda" if torch.cuda.is_available() else None,
            )

        recipe_label = f"recipe={self.config.recipe} transformer_only={self.config.transformer_only}"
        logger.info_rank0(f"ao_quantize applied to {len(targets)} target(s): {recipe_label}")
        context.extras.setdefault("ao_quantize", {})
        context.extras["ao_quantize"].update(
            recipe=self.config.recipe,
            transformer_only=self.config.transformer_only,
            n_targets=len(targets),
        )
