"""Composer integration: model wrapper, metrics, and the best-checkpoint callback."""

import logging
import math
from typing import Any, Literal

import torch
from composer.callbacks import CheckpointSaver
from composer.core import State, Timestamp
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.utils import get_save_filename
from torchmetrics import Metric

from src.models.common import LMOutput

log = logging.getLogger(__name__)


class Loss(Metric):
    """Mean of the per-batch losses."""

    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: LMOutput, target: Any = None) -> None:
        self.sum_loss += output["loss"]
        self.total_batches += 1

    def compute(self) -> torch.Tensor:
        return self.sum_loss / self.total_batches


class NLL(Metric):
    """Mean next-token negative log-likelihood over all loss tokens."""

    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mean_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: LMOutput, target: Any = None) -> None:
        nlls, mask = output["nlls"], output["loss_mask"]
        if nlls.numel() == 0:
            return
        self.mean_nll += (nlls * mask).sum()
        if self.weight.dtype != mask.dtype:
            self.weight = self.weight.to(mask.dtype)
        self.weight += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.mean_nll / self.weight


class BPD(NLL):
    """Bits per token."""

    def compute(self) -> torch.Tensor:
        return self.mean_nll / self.weight / math.log(2)


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_nll / self.weight)


def _metrics() -> dict[str, Metric]:
    return {m.__class__.__name__: m for m in (Loss(), NLL(), BPD(), Perplexity())}


class ComposerLM(ComposerModel):
    """Wrap a model whose forward takes the collated batch and returns `LMOutput`."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.train_metrics = _metrics()
        self.val_metrics = _metrics()

    def forward(self, batch: dict[str, torch.Tensor]) -> LMOutput:
        return self.model(**batch)

    def loss(self, outputs: LMOutput, batch: Any) -> torch.Tensor:
        return outputs["loss"]

    def get_metrics(self, is_train: bool = False) -> dict[str, Metric]:
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch: Any, outputs: LMOutput, metric: Metric) -> None:
        metric.update(outputs)


class SaveBestCheckpointing(CheckpointSaver):
    """Save a checkpoint after each evaluation that improves the monitored metric.

    Args:
        metric_to_monitor: ``"<train|eval>/<metric_name>"``, e.g. ``"eval/loss"``.
        mode: Whether lower (``"min"``) or higher (``"max"``) is better.
        args, kwargs: Passed to `CheckpointSaver`.
    """

    def __init__(
        self,
        metric_to_monitor: str,
        mode: Literal["min", "max"] = "min",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric_to_monitor = metric_to_monitor
        self.train_or_eval, _, self.metric_name = metric_to_monitor.partition("/")
        self.mode = mode
        self.best_value = None
        self.latest_filename = None

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return (
            value < self.best_value if self.mode == "min" else value > self.best_value
        )

    def eval_end(self, state: State, logger: Logger) -> None:
        metrics = (
            state.train_metric_values
            if self.train_or_eval == "train"
            else state.eval_metric_values
        )
        metrics = {
            key.lower(): (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in metrics.items()
        }
        if self.metric_name not in metrics:
            raise ValueError(f"Metric {self.metric_name} not found in {list(metrics)}")
        value = metrics[self.metric_name]
        if self._improved(value):
            save_filename = get_save_filename(
                state, self.filename.format(state, keep_placeholders=True)
            )
            log.info(
                f"Best {self.metric_name} attained: {value}. "
                f"Saving checkpoint to {save_filename}."
            )
            self.best_value = value
            self._save_checkpoint(state, logger)

    # Only save on improvement, never on the regular save interval.
    def batch_checkpoint(self, state: State, logger: Logger) -> None:
        pass

    def epoch_checkpoint(self, state: State, logger: Logger) -> None:
        pass

    def iteration_checkpoint(self, state: State, logger: Logger) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        # Store the best value alongside each checkpoint so it survives resumption.
        state_dict["all_saved_checkpoints_to_timestamp"] = [
            (
                filename,
                timestamp.state_dict(),
                {
                    "metric_to_monitor": self.metric_to_monitor,
                    "mode": self.mode,
                    "best_value": self.best_value,
                },
            )
            for filename, timestamp in self.all_saved_checkpoints_to_timestamp.items()
        ]
        return state_dict

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for filename, timestamp_state, metric in state.get(
            "all_saved_checkpoints_to_timestamp", []
        ):
            timestamp = Timestamp()
            timestamp.load_state_dict(timestamp_state)
            self.all_saved_checkpoints_to_timestamp[filename] = timestamp
            self.best_value = metric["best_value"]
