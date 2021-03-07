import random
import os
seed = 1
random.seed(1)

resolution = [256, 512, 1024]
data_root = '/media/nirvana/2.0TB-Disk/dataset/CelebA-HQ/'

imgs = [os.listdir(data_root + f'celeba-{reso}') for reso in resolution]
imgs = [set(img) for img in imgs]

for _ in range(len(imgs)):
    assert imgs[_] == imgs[0]

imgs = list(imgs[0])
random.shuffle(imgs)

def save_list(img_name_list, path):
    with open(path, 'w') as f:
        for img_name in img_name_list:
            f.write(f'{img_name}\n')

save_list(imgs[:27000], data_root + '27k_train_list.txt')
save_list(imgs[-3000:], data_root + '3k_val_list.txt')

save_list(imgs[:2700], data_root + 'tiny_2.7k_train_list.txt')
save_list(imgs[-300:], data_root + 'tiny_0.3k_val_list.txt')

