import deeplabcut.modelzoo.video_inference as modelzoo
import torch

def main():

    customized_pose_checkpoint = '/mediaPFM/data/datasets/final_datasets/DLCdev/benchmark/coco/pfm_pose_epoch_100_converted.pth'
    customized_detector_checkpoint = 'pfm_detector_10percent.pt'
    customized_model_config='/mediaPFM/data/datasets/final_datasets/DLCdev/benchmark/coco/pfm_detector/train/pytorch_config.yaml'


    modelzoo.video_inference_superanimal(
        videos=["/mediaPFM/data/datasets/macaque_monkey.mp4"],
        superanimal_name="superanimal_quadruped_hrnetw32",
        video_adapt=True,
        max_individuals=1,
        pseudo_threshold=0.1,
        bbox_threshold=0.9,
        detector_epochs=4,
        pose_epochs=4,
        pcutoff = 0.9,
        customized_pose_checkpoint = customized_pose_checkpoint,
        customized_detector_checkpoint = customized_detector_checkpoint,
        customized_model_config = customized_model_config
    )



if __name__ == "__main__":
    main()
