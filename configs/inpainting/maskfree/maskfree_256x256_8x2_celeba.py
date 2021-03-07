model = dict(
    type='MaskFreeInpaintor',
    encdec=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='NaiveEncoder',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True),
        decoder=dict(
            type='NaiveDecoder',
            out_channels=3,
            norm_cfg=dict(type='BN'))),
    disc=dict(
        type='MaskFreeDiscriminator',
        img_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 4 * 4,
            fc_out_channels=1024,
            num_convs=6,
            norm_cfg=dict(type='BN')),
        mask_disc_cfg=dict(
            type='MaskDiscriminator',
            encoder=dict(
                type='NaiveEncoder',
                norm_cfg=dict(type='BN', requires_grad=False),
                norm_eval=True),
            decoder=dict(
                type='NaiveDecoder',
                out_channels=1,

                norm_cfg=dict(type='BN'))),
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.001
    ),
    loss_gan_mask=dict(
        type='MaskL1Loss',
    ),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg16',
        layer_weights={
            '4': 1.,
            '9': 1.,
            '16': 1.,
        },
        perceptual_weight=0.05,
        style_weight=120,
        pretrained=('torchvision://vgg16')),
    loss_out_percep=True,
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=6.,
    ),
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.,
    ),
    loss_tv=dict(
        type='MaskedTVLoss',
        loss_weight=0.1,
    ),
    pretrained=None)

train_cfg = dict(disc_step=1)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

dataset_type = 'ImgInpaintingDataset'
input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(
        type='LoadMask',
        mask_mode='irregular',
        mask_config=dict(
            num_vertexes=(4, 10),
            max_angle=6.0,
            length_range=(20, 128),
            brush_width=(10, 45),
            area_ratio_range=(0.15, 0.65),
            img_shape=input_shape)),
    dict(
        type='Crop',
        keys=['gt_img'],
        crop_size=(384, 384),
        random_crop=True,
    ),
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='GetMaskedImage'),
    dict(
        type='Collect',
        keys=['gt_img', 'masked_img', 'mask'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask'])
]

test_pipeline = train_pipeline

data_root = 'data/CelebA-HQ/'

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type=dataset_type,
        ann_file=(data_root + '3k_val_list.txt'),
        data_prefix=data_root + 'celeba-1024',
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=(data_root + '3k_val_list.txt'),
        data_prefix=data_root + 'celeba-1024',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=(data_root + '3k_val_list.txt'),
        data_prefix=data_root + 'celeba-1024',
        pipeline=test_pipeline,
        test_mode=True))

optimizers = dict(generator=dict(type='Adam', lr=0.0004),
                  disc=dict(type='Adam', lr=0.00004))  # second stage training

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=50000)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', ),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='VisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=['gt_img', 'masked_img', 'fake_res',
                   'fake_img', 'mask_pred', 'mask'],
)

evaluation = dict(
    interval=50000,
    metric_dict=dict(l1=dict()),
)

total_iters = 300002
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 10000)]
exp_name = 'maskfree_256x256_8x2_celeba'
find_unused_parameters = False
