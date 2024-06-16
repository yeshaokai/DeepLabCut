#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import glob

import deeplabcut
from deeplabcut.pose_estimation_pytorch.apis.analyze_images import (
    superanimal_analyze_images,
)


if __name__ == "__main__":
    superanimal_name = "superanimal_quadruped"
    model_name = "hrnetw32"
    device = "cuda"
    max_individuals = 1

    customized_pose_checkpoint = '/mediaPFM/data/datasets/final_datasets/DLCdev/benchmark/coco/pfm_pose_epoch_100_converted.pth'
    customized_detector_checkpoint = '/mediaPFM/data/datasets/final_datasets/DLCdev/benchmark/coco/pfm_detector/train/snapshot-detector-010.pt'
    customized_model_config='/mediaPFM/data/datasets/final_datasets/DLCdev/benchmark/coco/pfm_detector/train/pytorch_config.yaml'
    
    ret = superanimal_analyze_images(
        superanimal_name,
        model_name,
        "/mediaPFM/data/datasets/monkey_images",
        max_individuals,
        "/mediaPFM/data/datasets/out_monkey_images",
        customized_pose_checkpoint = customized_pose_checkpoint,
        customized_detector_checkpoint = customized_detector_checkpoint,
        customized_model_config = customized_model_config        
    )
