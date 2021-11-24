
import pandas as pd
import cv2
import numpy as np
import torch
import json
import os,shutil
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt


def crop_image(dir_image_folder, dir_save):
    image_files = [f for f in os.listdir(dir_image_folder) if isfile(join(dir_image_folder, f))]
    image_files = sorted(image_files)
    total_image = 1
    root_dir = dir_save
    for filename in os.listdir(dir_save):
        file_path = os.path.join(dir_save, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass

    for j in image_files:
        dir_image = os.path.join(os.path.expanduser('~'), dir_image_folder, j)
        im = cv2.imread(dir_image, -1)
        im_height, im_width = im.shape[0], im.shape[1]

        if im_width % 300 == 0:
            nm_total_image = im_width / 300
            nm_total_image = int(nm_total_image)

            x, y = 0, 0
            for i in range(0, nm_total_image):
                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                if im_height < 300:
                    padding_amount = 300 - im_height
                    im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    print(total_image)
                    cv2.imwrite(dir_save, im_pad)
                else:

                    cv2.imwrite(dir_save, crop_im)
                x += 300
                total_image += 1
        else:
            nm_total_image = im_width / 300
            nm_total_image = int(nm_total_image) + 1

            x, y = 0, 0
            for i in range(1, nm_total_image):
                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                if im_height < 300:
                    padding_amount = 300 - im_height
                    im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    print(total_image)
                    cv2.imwrite(dir_save, im_pad)
                else:
                    print(total_image)
                    cv2.imwrite(dir_save, crop_im)
                x += 300
                total_image += 1

            if im_height < 300:
                total_image += 1
                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                x, y = im_width - 300, 0
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                padding_amount = 300 - im_height
                im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                print(total_image)
                cv2.imwrite(dir_save, im_pad)
            else:
                total_image += 1
                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                x, y = im_width - 300, 0
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                print(total_image)
                cv2.imwrite(dir_save, crop_im)

    print('\nCropped original images and created %d train and test images. Images has been saved to %s .' % (total_image,root_dir))


crop_image("/home/criuser/Desktop/Internship/Orginal_images", '/home/criuser/Desktop/Internship/Cropped_images')
def create_data_json( csv_folder_path):

    bbox_1 = pd.read_csv("/home/criuser/Desktop/Internship/1.csv")
    bbox_1 = bbox_1.loc[bbox_1['Slice'] != 0]
    bbox_1 = bbox_1[['BX', 'BY', 'Width', 'Height']]
    bbox_1 = {"xmin":bbox_1['BX'],"ymin":bbox_1['BY'],"xmax":bbox_1['BX'] + bbox_1['Width'], "ymax": bbox_1['BY'] + bbox_1['Height']}
    bbox_1 = pd.DataFrame(bbox_1)

    print(bbox_1.head(58))


# print(len(bbox_1))
# class EcoliBacteriaDataset(Dataset):
