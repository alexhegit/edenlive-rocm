"""Isolate the DataLoader layer (no model, no GPU). Run inside container."""
import time
from edenlive.core.config import apply_overrides, load_config
from edenlive.data.dataloader import build_dataloader, build_dataset
from edenlive.core.types import RuntimeContext
import torch

config = apply_overrides(
    load_config("edenlive/configs/wan22_5b_720p_few_step.yaml"),
    ["data.dataset.params.path=examples/samples/manifest.csv",
     "data.dataset.params.root=examples/samples"],
)
dsc = config["data"]
ds = build_dataset(dsc["dataset"])
print("dataset len:", len(ds), flush=True)

rc = RuntimeContext(device=torch.device("cuda", 0))
loader = build_dataloader(dsc, rc)
print("loader kwargs relevant:", "batch_size", loader.batch_size, "num_workers", loader.num_workers,
      "drop_last", getattr(loader, "_drop_last", "?"), flush=True)

count = 0
for i, batch in enumerate(loader):
    count += 1
    keys = list(batch.conditions.items)
    print(f"batch {i}: cond_keys={keys} text={type(batch.conditions.get('text').data).__name__}", flush=True)
    if count >= 2:
        break
print("TOTAL_BATCHES_SEEN:", count, flush=True)
print("DSCHECK_DONE", flush=True)
