#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

from deeplabcut.pose_estimation_pytorch.models.detectors.base import DETECTORS
from deeplabcut.pose_estimation_pytorch.models.detectors.torchvision import (
    TorchvisionDetectorAdaptor,
)


@DETECTORS.register_module
class SSDLite(TorchvisionDetectorAdaptor):
    """An SSD object detection model"""

    def __init__(
        self,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        pretrained: bool = False,
        box_score_thresh: float = 0.01,
    ) -> None:
        model_kwargs = None
        if pretrained:
            model_kwargs = dict(weights_backbone="IMAGENET1K_V2")

        super().__init__(
            model="ssdlite320_mobilenet_v3_large",
            weights=None,
            num_classes=2,
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            box_score_thresh=box_score_thresh,
            model_kwargs=model_kwargs,
        )
