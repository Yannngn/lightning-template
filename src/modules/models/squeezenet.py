from typing import Optional

from torch import Tensor, nn
from torchvision.models.squeezenet import SqueezeNet1_1_Weights, squeezenet1_1


class BaseSqueezeNetModule(nn.Module):
    def __init__(
        self,
        weights: Optional[SqueezeNet1_1_Weights] = None,
        num_classes: Optional[int] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.model = squeezenet1_1(
            weights=weights, num_classes=num_classes, dropout=dropout
        )

    def forward(self, x: Tensor, y=None) -> Tensor:
        return self.model(x)


class SqueezeNet1_1Module(BaseSqueezeNetModule):
    def __init__(
        self,
        finetune: bool = True,
        weights: Optional[SqueezeNet1_1_Weights] = None,
        num_classes: Optional[int] = None,
        trainable_feature_layers: Optional[int] = None,
        dropout: float = 0.5,
    ) -> None:
        if not finetune:
            super().__init__(
                weights=weights, num_classes=num_classes, dropout=dropout
            )
            return

        assert (
            num_classes is not None
        ), "if finetuning num_classes must be an integer"

        weights = weights or SqueezeNet1_1_Weights.DEFAULT

        super().__init__(weights=weights, dropout=dropout)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.num_classes = num_classes
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        if trainable_feature_layers:
            trainable_feature_layers *= -1
            for param in self.model.features[
                trainable_feature_layers:-1
            ].parameters():
                param.requires_grad = True
