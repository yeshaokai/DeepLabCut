det_model_name='superquadruped_faster_rcnn'
pose_model_name='superquadruped_tokenpose'
det_checkpoint='/mnt/md0/shaokai/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_quadruped/epoch_80.pth'
#pose_checkpoint='/mnt/md0/shaokai/integration/DeepLabCut/deeplabcut/superanimal_pytorch/third_party/work_dirs/hrnet_w32_quadruped_256x256_splitD/latest.pth'

pose_checkpoint='/mnt/md0/shaokai/integration/DeepLabCut/deeplabcut/superanimal_pytorch/third_party/work_dirs/Tokenpose_L_quadruped_256x256/latest.pth'

image_path='/mnt/md0/shaokai/many_dogs.jpeg'
device='cuda'
out_path='res.json'

python topdown_superanimal_img_inference.py $det_model_name $pose_model_name $det_checkpoint $pose_checkpoint $image_path --out $out_path --device $device