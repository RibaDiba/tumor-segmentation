"""Optional Weights & Biases writer. Activated when WANDB_PROJECT env is set.

Mirrors Detectron2 event-storage scalars (loss, mask loss, AP, IoU buckets, etc.)
to a W&B run. The merged training config is logged as the run's hyperparameters.
"""

import yaml
from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    def __init__(self, project: str, name: str, cfg):
        import wandb

        self._run = wandb.init(
            project=project,
            name=name,
            config=yaml.safe_load(cfg.dump()),
        )
        self._last = -1

    def write(self):
        storage = get_event_storage()
        if storage.iter == self._last:
            return
        scalars = {k: v.median(20) for k, v in storage.latest().items()}
        self._run.log(scalars, step=storage.iter)
        self._last = storage.iter

    def close(self):
        self._run.finish()
