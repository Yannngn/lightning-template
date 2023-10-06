from collections import OrderedDict
from typing import Dict, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from src.datamodules.components.transforms import TransformsWrapper


class AIVIDataModule(LightningDataModule):
    """Example of LightningDataModule for single dataset.

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def predict_dataloader(self):
            # return predict dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        """DataModule with standalone train, val and test dataloaders.

        Args:
            datasets (DictConfig): Datasets config.
            loaders (DictConfig): Loaders config.
            transforms (DictConfig): Transforms config.
        """

        super().__init__()
        self.cfg_datasets = datasets
        self.cfg_loaders = loaders
        self.transforms = TransformsWrapper(transforms)
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.predict_set: Dict[str, Dataset] = OrderedDict()

        self.collate_fn = hydra.utils.instantiate(
            self.cfg_loaders.get("collate_fn"), _partial_=True
        )

    def _get_dataset(
        self, stage: str, dataset_name: Optional[str] = None
    ) -> Dataset:
        self.transforms.set_mode(stage)
        cfg: DictConfig = self.cfg_datasets.get(stage)

        if cfg is None and stage == "valid":
            cfg = self.cfg_datasets.get("train")

        if dataset_name:
            cfg = cfg.get(dataset_name)

        return hydra.utils.instantiate(cfg, transforms=self.transforms)

    def _get_train_val_dataset(self) -> tuple[Dataset, Dataset]:
        dataset_train = self._get_dataset("train")
        dataset_val = self._get_dataset("valid")

        if self.cfg_datasets.get("valid") is not None:
            return dataset_train, dataset_val

        train_test_split = self.cfg_loaders.get("train_test_split")

        indices = torch.randperm(len(dataset_train)).tolist()  # type: ignore

        length = int(train_test_split * len(indices))

        dataset_train = Subset(dataset_train, indices[length:])
        dataset_val = Subset(dataset_val, indices[:length])

        return dataset_train, dataset_val

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        self.train_set, self.valid_set = self._get_train_val_dataset()
        self.test_set = self._get_dataset("test")

        # load predict datasets only if it exists in config
        if self.cfg_datasets.get("predict"):
            for dataset_name in self.cfg_datasets.get("predict").keys():
                self.predict_set[dataset_name] = self._get_dataset(
                    "predict", dataset_name=dataset_name
                )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        assert self.train_set is not None

        return DataLoader(
            self.train_set,
            **self.cfg_loaders.get("train"),
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.valid_set is not None
        return DataLoader(
            self.valid_set,
            **self.cfg_loaders.get("valid"),
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.test_set is not None
        return DataLoader(
            self.test_set,
            **self.cfg_loaders.get("test"),
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(
                    dataset,
                    **self.cfg_loaders.get("predict"),
                    collate_fn=self.collate_fn
                )
            )
        return loaders

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass
