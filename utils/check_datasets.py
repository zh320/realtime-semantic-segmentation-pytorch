import os
import shutil
import argparse
import random
random.seed(0)

import cv2
from tqdm import tqdm
import numpy as np
import labelme
from PIL import Image


def check_semantic_segmentation_datasets(datasets_path):
    out_path = datasets_path
    labels_path = os.path.join(out_path, 'labels')
    if not os.path.exists(labels_path):
        print('Error: %s not found' % (labels_path))
        return
    datasets_root = os.path.join(datasets_path, 'out')

    datasets_train = os.path.join(datasets_root, 'train')
    datasets_val = os.path.join(datasets_root, 'val')

    datasets_train_imgs = os.path.join(datasets_train, 'imgs')
    datasets_train_masks = os.path.join(datasets_train, 'masks')
    datasets_val_imgs = os.path.join(datasets_val, 'imgs')
    datasets_val_masks = os.path.join(datasets_val, 'masks')

    if os.path.exists(datasets_root):
        shutil.rmtree(datasets_root)

    for d in [datasets_root, datasets_train, datasets_val, datasets_train_imgs, datasets_train_masks, datasets_val_imgs, datasets_val_masks]:
        if not os.path.exists(d):
            os.makedirs(d)

    all_data = [i for i in os.listdir(labels_path) if os.path.splitext(i)[1] == '.json']
    print('all_data: ', len(all_data))

    print(all_data[:5])
    random.shuffle(all_data)
    print(all_data[:5])

    train_factor = 0.95
    train_num = round(train_factor * len(all_data))

    class_name_to_id = {'_background': 0}

    for i in tqdm(all_data):
        filename = os.path.splitext(os.path.basename(i))[0]
        label_file = labelme.LabelFile(filename=os.path.join(labels_path, i))
        for shape in label_file.shapes:
            if shape.get('shape_type', '') == 'polygon':
                if shape.get('label', 'None') not in class_name_to_id.keys():
                    class_name_to_id[shape.get('label', 'None')] = len(class_name_to_id)
    print(class_name_to_id)

    for i in tqdm(all_data[:train_num]):
        filename = os.path.splitext(os.path.basename(i))[0]
        label_file = labelme.LabelFile(filename=os.path.join(labels_path, i))
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        if img.ndim == 3:
            img = img[:, :, ::-1]
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        #cv2.imwrite(os.path.join(imgages_path, filename+'.png'), img)
        cv2.imencode('.png', img)[1].tofile(os.path.join(datasets_train_imgs, filename+'.png'))
        #cv2.imwrite(os.path.join(masks_path, filename+'.png'), lbl)
        cv2.imencode('.png', lbl)[1].tofile(os.path.join(datasets_train_masks, filename+'.png'))
    
    for i in tqdm(all_data[train_num:]):
        filename = os.path.splitext(os.path.basename(i))[0]
        label_file = labelme.LabelFile(filename=os.path.join(labels_path, i))
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        if img.ndim == 3:
            img = img[:, :, ::-1]
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        #cv2.imwrite(os.path.join(imgages_path, filename+'.png'), img)
        cv2.imencode('.png', img)[1].tofile(os.path.join(datasets_val_imgs, filename+'.png'))
        #cv2.imwrite(os.path.join(masks_path, filename+'.png'), lbl)
        cv2.imencode('.png', lbl)[1].tofile(os.path.join(datasets_val_masks, filename+'.png'))
    
    with open(os.path.join(datasets_root, 'data.yaml'), 'w+', encoding='UTF-8') as yaml_file:
        yaml_file.write('path: ')
        yaml_file.write(os.path.abspath(datasets_root))
        yaml_file.write('\n')
        
        yaml_file.write('names: \n')
        for i in range(len(class_name_to_id)):
            for k, v in class_name_to_id.items():
                if v == i:
                    yaml_file.write('  %d: %s\n' % (i, k))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_root', type=str, default="", help="path to datasets root dir.")
    
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    check_semantic_segmentation_datasets(opt.datasets_root)