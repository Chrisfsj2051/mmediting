_base_='L1_w6_pconv_256x256_stage1_8x1_celeba.py'
model = dict(
    loss_l1_hole=dict(
        type='SoftL1Loss',
        loss_weight=6.
    ))
