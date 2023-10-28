from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics
from src.modules.models.maskrcnn import BatchType


class ClassificationLitModule(BaseLitModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(network, optimizer, scheduler, logging, *args, **kwargs)
        self.loss = load_loss(network.get("loss"))
        self.output_activation = hydra.utils.instantiate(
            network.get("output_activation"), _partial_=True
        )

        main_metric, valid_metric_best, extra_metrics = load_metrics(
            network.get("metrics")
        )
        self.train_metric = main_metric.clone()
        self.train_extra_metrics = extra_metrics.clone(postfix="/train")

        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_extra_metrics = extra_metrics.clone(postfix="/valid")

        self.test_metric = main_metric.clone()
        self.test_extra_metrics = extra_metrics.clone(postfix="/test")

        self.save_hyperparameters(logger=False)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        x, y, _ = batch

        logits = self.forward(x)

        loss = self.loss(logits, y)

        preds = self.output_activation(logits)

        return loss, preds, y

    def predict_step(
        self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        images, _, _ = batch

        return self.model(images)

    def on_train_start(self) -> None:
        self.valid_metric_best.reset()

    def training_step(self, batch: BatchType, batch_idx: int) -> Dict[str, Any]:
        self.model.train()

        loss, preds, targets = self.model_step(batch, batch_idx)

        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,  # type: ignore
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/train",
            self.valid_metric,
            **self.logging_params,  # type: ignore
        )

        return {"loss": loss}

    def validation_step(self, batch: BatchType, batch_idx: int) -> Any:
        self.model.eval()

        loss, preds, targets = self.model_step(batch, batch_idx)

        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,  # type: ignore
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,  # type: ignore
        )

        self.valid_extra_metrics(preds, targets)
        self.log_dict(self.valid_extra_metrics, **self.logging_params)  # type: ignore
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()
        self.valid_metric_best(valid_metric)

        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params,  # type: ignore
        )

    def test_step(self, batch: BatchType, batch_idx: int) -> Any:
        self.model.eval()

        loss, preds, targets = self.model_step(batch, batch_idx)

        self.log(
            f"{self.loss.__class__.__name__}/test",
            loss,
            **self.logging_params,  # type: ignore
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,  # type: ignore
        )

        self.test_extra_metrics(preds, targets)
        self.log_dict(self.test_extra_metrics, **self.logging_params)  # type: ignore
        return {"loss": loss, "preds": preds, "targets": targets}


class MNISTLitModule(ClassificationLitModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        x, y = batch
        logits = self.forward(x["image"].permute(0, 3, 1, 2))
        loss = self.loss(logits, y)
        preds = self.output_activation(logits)

        return loss, preds, y

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        logits = self.forward(x["image"].permute(0, 3, 1, 2))
        preds = self.output_activation(logits)

        return {"logits": logits, "preds": preds, "targets": y}
