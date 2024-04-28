det_config=/mnt/md0/shaokai/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_quadruped.py
det_checkpoint=/mnt/md0/shaokai/mmdetection/detector_quadruped80k/latest.pth
pose_config=/mnt/md0/shaokai/DLC-ModelZoo/experiments/configs_finetune_superanimal/hrnet_w32_256x256_superquadruped.py
pose_checkpoint=/mnt/md0/shaokai/DLC-ModelZoo/SA_checkpoints_quadruped80k/hrnet_w32_quadruped_256x256_all/latest.pth

video_name=cat_piano

#python demo/top_down_video_demo_with_mmdet.py  $det_config \
#       $det_checkpoint \
#       $pose_config \
#       $pose_checkpoint \
#       --video-path ${video_name}.mp4 \
#       --out-video-root video_pred_${video_name}/ \
#       --kpt-thr 0.0 &&

#python videopseudo2annotation.py --video_result_path video_pred_${video_name}/${video_name}.mp4.json \
#       --out_root annotation_${video_name} \
#       --video_path ${video_name}.mp4 &&
    
#python tools/train.py  hrnet_pose_config_SA_quadruped.py --cfg-options data.train.ann_file=annotation_${video_name}/annotations/train.json data.train.img_prefix=annotation_${video_name}/images/ data.val.img_prefix=annotation_${video_name}/images/  data.val.ann_file=annotation_${video_name}/annotations/test.json total_epochs=4  lr_config.warmup_iters=1 optimizer.lr=5e-5 load_from=work_dirs/hrnet_w32_quadruped_256x256/latest.pth   --work-dir ${video_name}_adapted --freeze_BN &&

python demo/top_down_video_demo_with_mmdet.py  $det_config \
       $det_checkpoint \
       $pose_config \
       ${video_name}_adapted/latest.pth \
       --video-path ${video_name}.mp4 \
       --out-video-root ${video_name}_adapted_video_predict \
       --topk 1\
       --kpt-median-filter \
       --bbox-median-filter \
       --kpt-thr 0.3
