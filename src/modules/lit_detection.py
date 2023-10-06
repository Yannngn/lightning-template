from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics
from src.modules.models.maskrcnn import BatchType, EvalOutput, TrainOutput


class DetectionLitModule(BaseLitModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
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

    def on_train_start(self) -> None:
        self.valid_metric_best.reset()

    def training_step(
        self, batch: BatchType, batch_idx: int
    ) -> Dict[str, Any]:
        self.model.train()

        images, targets, _ = batch

        loss_dict: TrainOutput = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,  # type: ignore
        )

        return {"loss": loss}

    def validation_step(self, batch: BatchType, batch_idx: int) -> Any:
        self.model.eval()

        images, targets, _ = batch
        # images, targets, info = batch
        # _, image_names = zip(*info)

        outputs: EvalOutput = self.model(images)
        loss = (
            1
            - torch.mean(
                torch.cat([output["scores"] for output in outputs])
            ).item()
        )

        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,  # type: ignore
        )

        self.valid_metric(outputs, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,  # type: ignore
        )

        self.valid_extra_metrics(outputs, targets)
        self.log_dict(self.valid_extra_metrics, **self.logging_params)  # type: ignore
        return {"loss": loss, "preds": outputs, "targets": targets}

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

        images, targets, info = batch
        _, image_names = zip(*info)

        outputs: EvalOutput = self.model(images)

        loss = np.mean(
            [
                1 - score.item()
                for output in outputs
                for score in output["scores"]
            ]
        )

        self.log(
            f"{self.loss.__class__.__name__}/test",
            loss,
            **self.logging_params,  # type: ignore
        )

        self.test_metric(outputs, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,  # type: ignore
        )

        self.test_extra_metrics(outputs, targets)
        self.log_dict(self.test_extra_metrics, **self.logging_params)  # type: ignore
        return {"loss": loss, "preds": outputs, "targets": targets}

    def predict_step(
        self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        images, _, _ = batch
        return self.forward(images)
