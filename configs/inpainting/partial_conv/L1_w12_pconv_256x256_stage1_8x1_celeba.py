_base_='L1_w6_pconv_256x256_stage1_8x1_celeba.py'
model = dict(
    loss_l1_hole=dict(
        loss_weight=12.
    ))


