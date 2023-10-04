from typing import Dict, List

import torch
from torchmetrics import Metric


class IoU(Metric):
    def __init__(self, num_classes: int, dist_sync_on_step: bool = False, dist_reduce_fx: str = "sum") -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state(
            "intersection",
            default=torch.zeros((num_classes,)),
            dist_reduce_fx=dist_reduce_fx,
        )
        self.add_state(
            "union",
            default=torch.zeros((num_classes,)),
            dist_reduce_fx=dist_reduce_fx,
        )

    def update(self, preds: List[Dict[str, torch.Tensor]], target: List[Dict[str, torch.Tensor]]) -> None:
        for image_pred, image_target in zip(preds, target):
            masks = image_pred["masks"]
            pred_labels = image_pred["labels"]
            gts = image_target["masks"]
            gt_labels = image_target["labels"]

            for index in gt_labels.unique():
                masks = masks[pred_labels == index].detach().cpu()
                gts = gts[gt_labels == index].detach().cpu()

                intersection = gts.logical_and(masks)
                union = gts.logical_or(masks)

                self.intersection[index] += intersection.float().sum()  # type: ignore
                self.union[index] += union.float().sum()  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.intersection.sum() / self.union.sum()  # type: ignore
