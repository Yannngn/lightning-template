from typing import Dict, List

import cv2
import torch
from pytorch_lightning.utilities import FLOAT32_EPSILON
from torchmetrics import Metric


def get_iou(masks_1: torch.Tensor, masks_2: torch.Tensor) -> torch.Tensor:
    """Calculates the intersection over union (IoU) between two sets of masks.

    Args:
        masks_1: A tensor of masks with shape (N, H, W).
        masks_2: A tensor of masks with shape (M, H, W).

    Returns:
        A tensor of IoU scores with shape (N, M).
    """

    intersection = (
        (masks_1.unsqueeze(1) > 0) & (masks_2.unsqueeze(0) > 0)
    ).sum(dim=(2, 3))
    union = ((masks_1.unsqueeze(1) > 0) | (masks_2.unsqueeze(0) > 0)).sum(
        dim=(2, 3)
    ) + FLOAT32_EPSILON

    return intersection / union


class IoU(Metric):
    def __init__(
        self,
        num_classes: int,
        dist_sync_on_step: bool = False,
        dist_reduce_fx: str = "sum",
    ) -> None:
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

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        target: List[Dict[str, torch.Tensor]],
    ) -> None:
        for image_pred, image_target in zip(preds, target):
            masks = image_pred["masks"]
            pred_labels = image_pred["labels"].detach().cpu()
            gts = image_target["masks"]
            gt_labels = image_target["labels"].detach().cpu()

            for index in range(self.num_classes):
                masks = masks[pred_labels == index + 1].detach().cpu()
                gts = gts[gt_labels == index + 1].detach().cpu()

                intersection = gts.logical_and(masks)
                union = gts.logical_or(masks)

                self.intersection[index] += intersection.float().sum()  # type: ignore
                self.union[index] += union.float().sum()  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.intersection.sum() / self.union.sum()  # type: ignore


class DetectionCM(Metric):
    def __init__(
        self,
        num_classes: int,
        dist_sync_on_step: bool = False,
        dist_reduce_fx: str = "sum",
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state(
            "tp", default=torch.tensor(0.0), dist_reduce_fx=dist_reduce_fx
        )
        self.add_state(
            "fp", default=torch.tensor(0.0), dist_reduce_fx=dist_reduce_fx
        )
        self.add_state(
            "fn", default=torch.tensor(0.0), dist_reduce_fx=dist_reduce_fx
        )

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        tp = fn = fp = 0
        for image_pred, image_target in zip(predictions, targets):
            pred = image_pred["masks"]
            pred_labels = image_pred["labels"].detach().cpu()
            target = image_target["masks"]
            target_labels = image_target["labels"].detach().cpu()

            for index in range(1, self.num_classes + 1):
                masks = pred[pred_labels == index].detach().cpu()
                gts = target[target_labels == index].detach().cpu()

                ious = get_iou(masks, gts)

                tp += torch.sum(ious.max(dim=1)[1]).item()
                fn += torch.eq(ious.max(dim=0)[0], 0).sum().item()
                fp += (ious > 0).sum().item() - tp

        self.tp += tp  # type: ignore
        self.fn += fn  # type: ignore
        self.fp += fp  # type: ignore

    def compute(self) -> torch.Tensor:
        ...


class DetectionAccuracy(DetectionCM):
    def compute(self) -> torch.Tensor:
        return self.tp.float() / (self.tp + self.fp + self.fn + FLOAT32_EPSILON)  # type: ignore


class DetectionPrecision(DetectionCM):
    def compute(self) -> torch.Tensor:
        return self.tp.float() / (self.tp + self.fp + FLOAT32_EPSILON)  # type: ignore


class DetectionRecall(DetectionCM):
    def compute(self) -> torch.Tensor:
        return self.tp.float() / (self.tp + self.fn + FLOAT32_EPSILON)  # type: ignore


class DetectionF1(DetectionCM):
    def compute(self) -> torch.Tensor:
        precision = self.tp.float() / (self.tp + self.fp + FLOAT32_EPSILON)  # type: ignore
        recall = self.tp.float() / (self.tp + self.fn + FLOAT32_EPSILON)  # type: ignore

        return 2 * (precision * recall) / (precision + recall + FLOAT32_EPSILON)  # type: ignore


# def main():
#     def pseudo_masks(side, s):
#         square = torch.ones((s, s))

#         mask_1 = torch.zeros((side, side))
#         mask_2 = torch.zeros((side, side))
#         mask_3 = torch.zeros((side, side))

#         mask_1[0:s, 0:s] = square
#         mask_2[side - s : side, side - s : side] = square
#         mask_3[side // 2 - s // 2 : side // 2 + s // 2, side // 2 - s // 2 : side // 2 + s // 2] = square

#         return mask_1, mask_2, mask_3

#     gt1, gt2, _ = pseudo_masks(256, 100)

#     x1, x2, gt3 = pseudo_masks(256, 10)
#     m = torch.stack([gt1, gt2, x1, x2, torch.zeros((256, 256))])
#     g = torch.stack([gt1, gt2, gt3])

#     preds = [{"masks": m, "labels": torch.tensor([1, 1, 1, 1, 1])}]

#     gts = [{"masks": g, "labels": torch.tensor([1, 1, 1])}]

#     accuracy = DetectionAccuracy(1)
#     accuracy.update(preds, gts)
#     print(accuracy.compute())


# if __name__ == "__main__":
#     main()
