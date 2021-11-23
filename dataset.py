import pandas as pd
import cv2
import numpy as np
import torch
import json
import os
from PIL import Image

def crop_image(dir_im,dir_save):
    im = cv2.imread(dir_im)
    im_height, im_width = im.shape[0], im.shape[1]
    root_dir = dir_save
    if im_width%300 == 0:
        nm_total_image = im_width/300
        nm_total_image = int(nm_total_image)
        for i in range(0, nm_total_image):
            name = "P"+str(i) + ".tif"
            dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
            x, y = 0, 0
            width, height = 300, 300
            crop_im = im[y:y + height, x:x + width]
            if im_height < 300:
                padding_amount = 300 - im_height
                im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                cv2.imwrite(dir_save, im_pad)
            else:
                cv2.imwrite(dir_save, crop_im)
            x += 300
    else:
        nm_total_image = im_width/300
        nm_total_image = int(nm_total_image)+1
        for i in range(1,nm_total_image):
            name = "P" + str(i) + ".tif"
            dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
            x, y = 0, 0
            width, height = 300, 300
            crop_im = im[y:y + height, x:x + width]
            if im_height < 300:
                padding_amount = 300 - im_height
                im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                cv2.imwrite(dir_save, im_pad)
            else:
                cv2.imwrite(dir_save, crop_im)
            x += 300

        if im_height < 300:

            name = "P" + str(nm_total_image) + ".tif"
            dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
            x, y = im_width-300, 0
            width, height = 300, 300
            crop_im = im[y:y + height, x:x + width]
            padding_amount = 300 - im_height
            im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            cv2.imwrite(dir_save, im_pad)
        else:
            name = "P" + str(nm_total_image) + ".tif"
            dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
            x, y = im_width - 300, 0
            width, height = 300, 300
            crop_im = im[y:y + height, x:x + width]
            cv2.imwrite(dir_save, crop_im)


crop_image('/home/criuser/Desktop/Internship/Original-s20-ss-382-roi-24.tif','Desktop/Internship/Cropped_20' )


#bbox_1 = pd.read_csv("/home/criuser/Desktop/Internship/1.csv")

#bbox_1 = bbox_1.loc[bbox_1['Slice'] != 0]


#print(len(bbox_1))
#class EcoliBacteriaDataset(Dataset):
