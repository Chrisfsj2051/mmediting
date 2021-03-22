import os

# folder_name = 'inpainting/global_local/'
folder_name = 'inpainting/partial_conv/'
config_path = f'configs/{folder_name}/'

for i, file_name in enumerate(os.listdir(config_path)):
    # if 'id118' not in file_name:
    #     continue
    gpu_num = 8
    full_path = config_path + file_name
    config_prefix = file_name[:-3]
    outfile_name = config_prefix + '.txt'
    if i % 3 == 0:
        parti = 'mediaf'
    else:
        parti = 'mediaf1'
    parti = 'mediaa'
    # parti = 'mediaf1'
    # parti = 'MediaA'
    # parti = 'pat_mars1'
    print('nohup bash tools/fast_slurm_train.sh '
          '%s %s %s %s %d > log/%s/%s &' %
          (folder_name, file_name, parti, 'python', gpu_num, folder_name,
           outfile_name))
