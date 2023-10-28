from typing import Optional, Tuple

from torch import Tensor, nn
from torchvision.models.resnet import (
    ResNet101_Weights,
    ResNeXt101_32X8D_Weights,
    resnet101,
    resnext101_32x8d,
)


class BaseResNetModule(nn.Module):
    def __init__(
        self,
        weights: Optional[ResNet101_Weights] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = resnet101(weights=weights, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResNetModule(BaseResNetModule):
    def __init__(
        self,
        finetune: bool = True,
        weights: Optional[ResNet101_Weights] = None,
        num_classes: Optional[int] = None,
        classifier_dims: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not finetune:
            super().__init__(weights=weights, num_classes=num_classes)
            return

        assert (
            num_classes is not None
        ), "if finetuning num_classes must be an integer"

        weights = ResNet101_Weights.DEFAULT or weights
        classifier_dims = classifier_dims or (4096, 4096)

        super().__init__(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # nn.init.xavier_uniform_(self.model.fc.weight)

        for param in self.model.fc.parameters():
            param.requires_grad = True
