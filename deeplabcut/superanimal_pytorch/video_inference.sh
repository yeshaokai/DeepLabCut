export CUDA_VISIBLE_DEVICES=2
det_config=/mnt/md0/shaokai/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_quadruped.py
det_checkpoint=det_config_SA_quadruped.py
pose_config=configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/allquadruped/hrnet_w32_quadruped_256x256.py
pose_checkpoint=horse10_zeroshot_semi_shuffle1/latest.pth
video_path=/mnt/md0/shaokai/DLC-ModelZoo/data/all_quadruped/horse10/labeled-data/Sample15.mp4


python demo/top_down_video_demo_with_mmdet.py  $det_config \
       $det_checkpoint \
       $pose_config \
       $pose_checkpoint \
       --video-path $video_path \
       --out-video-root randomtest \
       --kpt-thr 0.0

