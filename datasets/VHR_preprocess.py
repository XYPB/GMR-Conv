import os
import numpy as np 
from PIL import Image
from glob import glob
from collections import Counter

def crop_instance(img_path, gt_path, dest_dir, target_size=(64, 64)):
    img_path = img_path.replace('\\', '/')
    im = Image.open(img_path)
    im = np.array(im)
    labels = []
    with open(gt_path, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            line = line.strip()
            if line == '':
                continue
            line = line.replace('(', '').replace(')', '')
            x1, y1, x2, y2, label = [int(num) for num in line.split(',')]

            sub_im = im[y1:y2, x1:x2, :]
            sub_im = Image.fromarray(sub_im).resize(target_size)
            img_idx = img_path.split('/')[-1].replace('.jpg', '')
            dest = os.path.join(dest_dir, f'{img_idx}_{i+1:03d}_{label}.png')
            sub_im.save(dest)
            labels.append(label)
    return labels



if __name__ == '__main__':
    input_dir = 'data/NWPU_VHR-10_dataset/positive_image_set'
    gt_dir = 'data/NWPU_VHR-10_dataset/ground_truth'
    dest_dir = 'data/NWPU_VHR-10_dataset/instance_image_set'
    os.makedirs(dest_dir, exist_ok=True)
    images = sorted(glob(os.path.join(input_dir, '*.jpg')))
    gts = sorted(glob(os.path.join(gt_dir, '*.txt')))

    labels = []
    for im_path, gt_path in zip(images, gts):
        labels += crop_instance(im_path, gt_path, dest_dir)
    print(Counter(labels))




