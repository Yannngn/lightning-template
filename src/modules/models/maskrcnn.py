from typing import Any, Dict, List, Optional, Tuple, TypeAlias

from torch import Tensor, nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNNPredictor,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.resnet import ResNet50_Weights

BatchType: TypeAlias = Tuple[List[Tensor], List[Dict[str, Tensor]], List[Tuple[Any, ...]]]

EvalOutput: TypeAlias = List[Dict[str, Tensor]]

TrainOutput: TypeAlias = Dict[str, Tensor]


class BaseMaskRCNNV2Module(nn.Module):
    def __init__(
        self,
        weights: Optional[MaskRCNN_ResNet50_FPN_V2_Weights] = None,
        num_classes: Optional[int] = None,
        backbone_weights: Optional[ResNet50_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            num_classes=num_classes,
            backbone_weights=backbone_weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    def forward(self, x: Tensor) -> list[dict[str, Tensor]]:
        return self.model(x)


class MaskRCNNV2Module(BaseMaskRCNNV2Module):
    def __init__(
        self,
        finetune: bool = True,
        weights: Optional[MaskRCNN_ResNet50_FPN_V2_Weights] = None,
        num_classes: Optional[int] = None,
        backbone_weights: Optional[ResNet50_Weights] = None,
        trainable_backbone_layers: Optional[int] = None,
    ) -> None:
        if not finetune:
            super().__init__(
                weights=weights,
                num_classes=num_classes,
                backbone_weights=backbone_weights,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            return

        assert num_classes is not None, "if finetuning num_classes must be an integer"

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if weights is None else weights

        super().__init__(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # type: ignore

        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore

        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask, dim_reduced=256, num_classes=num_classes
        )
