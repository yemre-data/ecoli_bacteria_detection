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
    print("ok")
    if im_width%300 == 0:
        nm_total_image = im_width/300
        nm_total_image = int(nm_total_image)
        for i in range(0, nm_total_image):
            dir_save = dir_save + "/" + str(i) + "_.tif"
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
        for i in range(0,nm_total_image):
            dir_save = dir_save + "/" + str(i) + "_.tif"
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
            dir_save = dir_save + "/_last.tif"
            x, y = im_width-300, 0
            width, height = 300, 300
            crop_im = im[y:y + height, x:x + width]
            padding_amount = 300 - im_height
            im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            cv2.imwrite(dir_save, im_pad)
        else:
            dir_save = dir_save + "/_last.tif"
            x, y = im_width - 300, 0
            width, height = 300, 300
            crop_im = im[y:y + height, x:x + width]
            cv2.imwrite(dir_save, crop_im)

#first
crop_image("/home/criuser/Desktop/Internship/Original-s1-ss-286-roi-24.tif","home/criuser/Desktop/Internship/Cropped_1")



#bbox_1 = pd.read_csv("/home/criuser/Desktop/Internship/1.csv")

#bbox_1 = bbox_1.loc[bbox_1['Slice'] != 0]


#print(len(bbox_1))
#class EcoliBacteriaDataset(Dataset):
