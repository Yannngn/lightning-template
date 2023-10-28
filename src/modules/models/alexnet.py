from typing import Optional, Tuple

from torch import Tensor, nn
from torchvision.models.alexnet import AlexNet_Weights, alexnet


class BaseAlexNetModule(nn.Module):
    def __init__(
        self,
        weights: Optional[AlexNet_Weights] = None,
        num_classes: Optional[int] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.model = alexnet(weights=weights, num_classes=num_classes, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class AlexNetModule(BaseAlexNetModule):
    def __init__(
        self,
        finetune: bool = True,
        weights: Optional[AlexNet_Weights] = None,
        num_classes: Optional[int] = None,
        classifier_dims: Optional[Tuple[int, int]] = None,
        dropout: float = 0.5,
    ) -> None:
        if not finetune:
            super().__init__(
                weights=weights,
                num_classes=num_classes,
                dropout=dropout,
            )
            return

        assert num_classes is not None, "if finetuning num_classes must be an integer"

        weights = AlexNet_Weights.DEFAULT or weights
        classifier_dims = classifier_dims or (4096, 4096)

        super().__init__(
            weights=weights,
            dropout=dropout,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, classifier_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_dims[1], num_classes),
        )

        for param in self.model.classifier.parameters():
            param.requires_grad = True
