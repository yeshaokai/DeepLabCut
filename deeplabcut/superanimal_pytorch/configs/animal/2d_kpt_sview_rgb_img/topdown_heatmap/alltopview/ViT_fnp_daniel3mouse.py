log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='mAP', save_best='AP')
kpts_num = 27
norm_cfg = dict(type='LN', requires_grad=True)

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip= dict(max_norm=0.003))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[350, 400])
total_epochs = 400
log_config = dict(
    interval=1,
    hooks=[
        # dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='vit_pose_estimation',
                 name='ViT_alltopview')
             )
    ])

channel_cfg = dict(
    num_output_channels=kpts_num,
    dataset_joints=kpts_num,
    dataset_channel=range(kpts_num),
    inference_channel=range(kpts_num)
)

# model settings
model = dict(
    type='TopDown',
    pretrained='/home/epfl_studenttemp/DeepLabCutv2/deeplabcut/cores/pose/notebook/deit_base_patch16_224-b5f2ef4d_.pth',
    backbone=dict(
        type='VisionTransformer',
        patch_size=16,
        use_fpn=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    ),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=kpts_num,
        norm_cfg=norm_cfg,
        use_residual=False,
        num_outs=5),
    keypoint_head=dict(
        type='TopdownHeatmapMultiHead',
        loss_keypoint=dict(type='JointsMSELoss',
                           use_target_weight=True,
                           gradient_masking = True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)
data_cfg = dict(
    image_size=[224, 224],
    heatmap_size=[56, 56],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/daniel3mouse_val_bbox_AP16.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', #'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            #'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/daniel3mouse'
dataset_info = 'data/all_topview_70-D-O/dataset.json'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type='TopDownDLCGenericDataset',
        ann_file=f'{data_root}/annotations/train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info,
    ),
    val=dict(
        type='TopDownDLCGenericDataset',
        ann_file=f'{data_root}/annotations/test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info
    ),
    test=dict(
        type='TopDownDLCGenericDataset',
        ann_file=f'{data_root}/annotations/test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info
    ),
)