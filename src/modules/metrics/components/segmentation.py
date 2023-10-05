import torch
from torchmetrics import Metric


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

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        res = preds.argmax(dim=1)
        for index in range(self.num_classes):
            gt = target.detach().cpu() == index
            preds = res == index

            intersection = gt.logical_and(preds.detach().cpu())
            union = gt.logical_or(preds.detach().cpu())

            self.intersection[index] += intersection.float().sum()  # type: ignore
            self.union[index] += union.float().sum()  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.intersection.sum() / self.union.sum()  # type: ignore
