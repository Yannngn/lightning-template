from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


class BaseLitModule(LightningModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """BaseLightningModule.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(*args, **kwargs)
        self.model: torch.nn.Module = hydra.utils.instantiate(network.model)
        self.optimizer_params = optimizer
        self.scheduler_params = scheduler
        self.logging_params = logging

    def forward(self, x: Any) -> Any:
        return self.model.forward(x)

    def configure_optimizers(self) -> Any:
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.optimizer_params, params=self.parameters(), _convert_="partial"
        )

        if params := self.scheduler_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.instantiate(
                params,
                optimizer=optimizer,
                _convert_="partial",
            )

            lr_scheduler_dict = {"scheduler": scheduler}

            if extra_params := self.scheduler_params.get("extras"):
                for key, value in extra_params.items():
                    lr_scheduler_dict[key] = value

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

        return {"optimizer": optimizer}
