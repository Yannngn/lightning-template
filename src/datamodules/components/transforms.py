from typing import Any

import albumentations
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        self.mode = "train"

        bbox_params = None
        if bbox_params := transforms_cfg.get("bbox_params"):
            bbox_params = hydra.utils.instantiate(bbox_params)

        # train augmentations
        train_aug = []
        if transforms_cfg.train.get("order"):
            for aug_name in transforms_cfg.train.get("order"):
                aug = hydra.utils.instantiate(
                    transforms_cfg.train.get(aug_name), _convert_="object"
                )
                train_aug.append(aug)
        self.train_aug = albumentations.Compose(
            train_aug, bbox_params=bbox_params
        )

        # valid, test and predict augmentations
        valid_test_predict_aug = []
        if transforms_cfg.valid_test_predict.get("order"):
            for aug_name in transforms_cfg.valid_test_predict.get("order"):
                aug = hydra.utils.instantiate(
                    transforms_cfg.valid_test_predict.get(aug_name),
                    _convert_="object",
                )
                valid_test_predict_aug.append(aug)
        self.valid_test_predict_aug = albumentations.Compose(
            valid_test_predict_aug, bbox_params=bbox_params
        )

    def set_mode(self, mode: str) -> None:
        """Set `__call__` mode.

        Args:
            mode (str): Applying mode.
        """

        self.mode = mode

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        That module has two modes: `train` and `valid_test_predict`.

        Args:
            image (Any): Input image.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """

        if isinstance(image, Image.Image):
            image = np.asarray(image)

        if self.mode == "train":
            return self.train_aug(image=image, **kwargs)

        return self.valid_test_predict_aug(image=image, **kwargs)
