from typing import Dict, List

import torch
from pytorch_lightning.utilities import FLOAT32_EPSILON
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

            for index in range(self.num_classes):
                masks = masks[pred_labels == index].detach().cpu()
                gts = gts[gt_labels == index].detach().cpu()

                intersection = gts.logical_and(masks)
                union = gts.logical_or(masks)

                self.intersection[index] += intersection.float().sum()  # type: ignore
                self.union[index] += union.float().sum()  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.intersection.sum() / self.union.sum()  # type: ignore


class DetectionAccuracy(Metric):
    def __init__(self, num_classes: int, dist_sync_on_step: bool = False, dist_reduce_fx: str = "sum") -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx=dist_reduce_fx)

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
        for image_pred, image_target in zip(predictions, targets):
            preds = image_pred["masks"]
            pred_labels = image_pred["labels"]
            gts = image_target["masks"]
            gt_labels = image_target["labels"]

            for index in range(self.num_classes + 1):
                print(preds.shape)
                print(pred_labels == index)
                masks = preds[pred_labels == index].detach().cpu()
                gt = gts[gt_labels == index].detach().cpu()

                ious = get_iou(masks, gt)
                max_iou, _ = ious.max(dim=1)
                print(ious)
                tp = masks[max_iou > 0].numel()
                self.tp += tp  # type: ignore
                self.fp += masks[ious > 0].sub(tp).numel()  # type: ignore

                max_iou, _ = ious.max(dim=2)

                self.fn += gt[max_iou == 0].numel()  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.tp.float() / (self.tp + self.fp + self.fn + FLOAT32_EPSILON)  # type: ignore


def get_iou(masks_1: torch.Tensor, masks_2: torch.Tensor) -> torch.Tensor:
    intersection = masks_1.logical_and(masks_2)
    union = masks_1.logical_or(masks_2)

    return intersection.div(union)


def main():
    def pseudo_masks(side, s):
        # Create a new PyTorch tensor of the desired size and shape.
        mask = torch.zeros((side, side))

        square = torch.ones((s, s))

        mask_1 = mask
        mask_2 = mask

        mask_1[0:s, 0:s] = square
        mask_2[side - s : side, side - s : side] = square

        return mask_1, mask_2

    gt1, gt2 = pseudo_masks(256, 100)
    x1, x2 = pseudo_masks(256, 10)
    m = torch.stack([gt1, gt2, x1, x2, torch.zeros((256, 256))])
    g = torch.stack([gt1, gt2])
    print(m.shape)
    preds = [
        {
            "masks": m,
            "labels": torch.tensor([1, 1, 1, 1, 1]),
        }
    ]

    gts = [{"masks": g, "labels": torch.tensor([1, 1])}]

    accuracy = DetectionAccuracy(1)
    accuracy.update(preds, gts)


if __name__ == "__main__":
    main()
