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

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

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

    def validation_epoch_end(self, outputs: List[Any]) -> None:
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

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def predict_step(
        self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        images, _, _ = batch
        return self.forward(images)


'''class DetectionModule__(pl.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Iterator[Parameter]], optim.Optimizer],
        lr_scheduler: Callable[[optim.Optimizer], optim.lr_scheduler.LRScheduler],
        metrics: CustomMetric,
        num_classes: int,
        hidden_layer: int | None = None,
        transfer: bool = True,
        monitor: str = "train_loss",
        interval: str = "epoch",
        save_dir: str = "./data/results/detections/",
    ) -> None:
        super().__init__()
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.num_classes = num_classes
        self.hidden_layer = hidden_layer
        self.transfer = transfer
        self.lr_monitor = monitor
        self.lr_interval = interval
        self.save_dir = save_dir

        self.model = self.get_model()

        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")

    def configure_optimizers(self) -> Any:
        self.optimizer = self.__optimizer(self.model.parameters())
        self.lr_scheduler = self.__lr_scheduler(self.optimizer)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": self.lr_monitor,
                "interval": self.lr_interval,
            },
        }

    def training_step(
        self, batch: tuple[list[Tensor], list[dict[str, Tensor]], list[tuple[int, str]]], batch_idx
    ) -> dict[str, int]:
        self.model.train()
        images, targets, _ = batch

        loss_dict: dict[str, Any] = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log(
            "train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(targets)
        )

        return {"loss": losses}

    def validation_step(
        self, batch: tuple[list[Tensor], list[dict[str, Tensor]], list[tuple[Any]]], batch_idx
    ) -> dict[str, Any]:
        self.model.eval()

        images, targets, info = batch
        _, image_names = zip(*info)

        outputs: list[dict[str, Tensor]] = self.model(images)

        if isinstance(self.logger, CustomCometLogger):
            # use cpu to free gpu vram, only 5 images in val
            cpu_images = [image.detach().cpu() for image in images][:5]
            cpu_outputs = [
                {k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in output.items()} for output in outputs
            ][:5]

            self.logger.custom_log_batch(image_names, cpu_images, cpu_outputs)  # type: ignore

        self.val_metrics.reset()
        self.val_metrics.update(outputs, targets)
        bbox_dict = self.val_metrics.compute()

        self.log_dict(bbox_dict, on_step=True, on_epoch=True, logger=True, batch_size=len(targets))

        return bbox_dict

    def on_test_epoch_start(self) -> None:
        if isinstance(self.logger, CustomCometLogger):
            self.logger.create_stats_data()

    def test_step(
        self, batch: tuple[list[Tensor], list[dict[str, Tensor]], list[tuple[Any]]], batch_idx: int
    ) -> dict[str, Any]:
        self.model.eval()

        images, targets, info = batch
        _, image_names = zip(*info)
        outputs: list[dict[str, Tensor]] = self.model(images)

        if isinstance(self.logger, CustomCometLogger):
            # use cpu to free gpu vram
            cpu_images = [image.detach().cpu() for image in images]
            cpu_targets = [
                {k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in target.items()} for target in targets
            ]
            cpu_outputs = [
                {k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in output.items()} for output in outputs
            ]

            self.logger.custom_log_batch(image_names, cpu_images, cpu_outputs)  # type: ignore
            self.logger.update_stats_data(image_names, cpu_targets, cpu_outputs)  # type: ignore

        self.test_metrics.reset()
        self.test_metrics.update(outputs, targets)
        bbox_dict = self.test_metrics.compute()
        self.log_dict(bbox_dict, on_step=True, on_epoch=True, logger=True, batch_size=len(targets))

        return bbox_dict

    def on_test_epoch_end(self) -> None:
        if isinstance(self.logger, CustomCometLogger):
            self.logger.log_stats_data()

    def predict_step(
        self, batch: tuple[list[Tensor], list[dict[str, Tensor]], list[tuple[Any]]], batch_idx: int
    ) -> None:
        """Use to save locally"""
        self.model.eval()
        images, _, info = batch
        _, image_names = zip(*info)
        outputs: list[dict[str, Tensor]] = self.model(images)

        cpu_images = [image.detach().cpu() for image in images]
        cpu_outputs = [
            {k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in output.items()} for output in outputs
        ]

        torch_save_predictions(
            images=cpu_images, predictions=cpu_outputs, image_names=image_names, output_dir=self.save_dir  # type: ignore
        )
'''
