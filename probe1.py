"""Step-by-step probe of the offline path. Run inside the container."""
import faulthandler, os, sys, time, traceback
from pathlib import Path
faulthandler.enable()
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.1f}] {msg}", flush=True)

from edenlive.core.config import apply_overrides, load_config
from edenlive.core.factory import build_model_adapter, build_runtime_plugins
from edenlive.core.types import RuntimeContext
from edenlive.runtime.precision import attach_runtime_precision, resolve_torch_dtype
from edenlive.inference.offline import _init_distributed, _condition_text_items, _is_artifact_writer

config = apply_overrides(
    load_config("edenlive/configs/wan22_5b_720p_few_step.yaml"),
    ["data.dataset.params.path=examples/samples/manifest.csv",
     "data.dataset.params.root=examples/samples",
     "inference.max_batches=1"],
)
log("config loaded")

# 1) dataset alone
from edenlive.data import build_dataloader
ds_config = config["data"]
from edenlive.data.dataloader import build_dataset
ds = build_dataset(ds_config["dataset"])
log(f"dataset built: len={len(ds)}")
batch0 = ds[0]
log(f"batch0 keys={list(batch0['conditions'].keys())} meta_id={batch0['metadata'].get('id')}")
import torch
for k, v in batch0["conditions"].items():
    try:
        shp = tuple(v.shape)
    except Exception:
        shp = type(v).__name__
    log(f"  cond {k}: {shp}")

# 2) full pipeline
device, rank, world = _init_distributed()
log(f"dist: device={device} rank={rank} world={world}")
rc = RuntimeContext(device=device)
rc.extras["mode"] = "inference"
compute = resolve_torch_dtype("bfloat16")
attach_runtime_precision(rc, compute_dtype=compute, param_dtype=compute)
model = build_model_adapter(config)
plugins = build_runtime_plugins(config)
for p in plugins:
    rc = p.attach(model, rc)
model.load_components(rc)
log("model loaded")
for p in sorted(plugins, key=lambda p: {"torch_compile":0,"sequence_parallel":1,"fsdp2":2}.get(p.name,1)):
    p.prepare_model(model, rc)
log("prepared")

from edenlive.inference.engine import InferenceEngine
from edenlive.core.types import ConditionBundle, ModalityPayload
from edenlive.data.dataloader import collate_to_batch
from edenlive.inference.io import save_generation_artifacts
engine = InferenceEngine(model, runtime_context=rc, config=config["inference"])
log("engine built")
log(f"artifact_writer={_is_artifact_writer(rc)}")
torch.cuda.synchronize()
batch = collate_to_batch([ds[0]])
t_gen = time.time()
prompts = _condition_text_items(batch, int(config["inference"].get("num_samples", 1)))
log(f"prompts={len(prompts)}: {str(prompts)[:80]!r}")
from edenlive.inference.request import GenerationRequest
conditions = ConditionBundle(
    items={**batch.conditions.items, "text": ModalityPayload("text", prompts)},
    raw=dict(batch.conditions.raw),
)
params = dict(config["inference"])
proxy = conditions.get("video_context")
if proxy is not None and "num_frames" in proxy.metadata:
    params["num_frames"] = int(proxy.metadata["num_frames"])
    log(f"num_frames={params['num_frames']}")
chunk_prompts = conditions.get("chunk_prompts")
if chunk_prompts is not None:
    params["chunk_prompts"] = chunk_prompts.data[0]
    log(f"chunk_prompts items={len(params['chunk_prompts'])}")
request = GenerationRequest(
    conditions=conditions,
    strategy=str(config["inference"].get("strategy", "diffusion")),
    outputs=tuple(config["inference"].get("outputs", ["video"])),
    params=params,
    metadata={"source": "probe", "batch_idx": 0},
)
with torch.no_grad():
    output = engine.generate(request)
torch.cuda.synchronize()
log(f"GENERATE OK in {time.time()-t_gen:.1f}s")
try:
    log(f"output type={type(output).__name__} fields={ [a for a in dir(output) if not a.startswith('_')] }")
except Exception:
    traceback.print_exc()
save_root = Path("outputs/probe")
save_root.mkdir(parents=True, exist_ok=True)
save_generation_artifacts(
    batch=batch, output=output, prompts=prompts, names=["probe_rocm"],
    step="base", output_dir=save_root, config=config["inference"],
)
log("SAVED OK")
for root, _, files in os.walk(save_root):
    for f in files:
        p = os.path.join(root, f)
        log(f"  {p} {os.path.getsize(p)} bytes")
log("PROBE DONE")
