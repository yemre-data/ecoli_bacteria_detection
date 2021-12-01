
import pandas as pd
import cv2
import numpy as np
import torch
import json
import os,shutil
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt


def create_images_and_bbox_list(dir_image_folder,dir_csv_folder, dir_save):
    image_files = [f for f in os.listdir(dir_image_folder) if isfile(join(dir_image_folder, f))]
    bbox_files = [f for f in os.listdir(dir_csv_folder) if isfile(join(dir_csv_folder, f))]
    image_files = sorted(image_files)
    bbox_files = sorted(bbox_files)
    total_image = 1
    root_dir = dir_save
    images = list()
    objects = list()
    n_box = 0

    for filename in os.listdir(dir_save):
        file_path = os.path.join(dir_save, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass

    for j,b in zip(image_files,bbox_files):
        dir_image = os.path.join(os.path.expanduser('~'), dir_image_folder, j)
        dir_bbox = os.path.join(os.path.expanduser('~'), dir_csv_folder, b)

        im = cv2.imread(dir_image, -1)
        im_height, im_width = im.shape[0], im.shape[1]
        bbox_df = pd.read_csv(dir_bbox)
        bbox_df = bbox_df.loc[bbox_df['Slice'] == 1]
        bbox_df = bbox_df[['BX', 'BY', 'Width', 'Height']]
        bbox_df = {"xmin": bbox_df['BX'], "ymin": bbox_df['BY'], "xmax": bbox_df['BX'] + bbox_df['Width'],
                  "ymax": bbox_df['BY'] + bbox_df['Height']}
        bbox_df = pd.DataFrame(bbox_df)
        if im_width % 300 == 0:
            nm_total_image = im_width / 300
            nm_total_image = int(nm_total_image)

            x, y = 0, 0
            for i in range(0, nm_total_image):
                box_temp = []
                label_tem = []
                for index,row in bbox_df.iterrows():
                    if x <= row["xmin"] <= x+300 and row["xmin"] < nm_total_image*300:
                        if im_height > 300:
                            box_temp.append([int(row["xmin"]-x),int(row["ymin"]),int(row["xmax"]-x),int(300)])
                            label_tem.append(1)
                            n_box += 1
                        else:
                            box_temp.append(
                                [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(row["ymax"])])
                            label_tem.append(1)
                            n_box += 1
                objects.append({"boxes":box_temp,"labels":label_tem})

                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                if im_height < 300:
                    padding_amount = 300 - im_height
                    im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    cv2.imwrite(dir_save, im_pad)
                    images.append(dir_save)
                    total_image += 1
                else:
                    cv2.imwrite(dir_save, crop_im)
                    images.append(dir_save)
                    total_image += 1
                x += 300

        else:
            nm_total_image = im_width / 300
            nm_total_image = int(nm_total_image) + 1

            x, y = 0, 0
            for i in range(1, nm_total_image):
                box_temp = []
                label_tem = []
                for index, row in bbox_df.iterrows():
                    if x <= row["xmin"] <= x+300 and row["xmin"] < nm_total_image*300:
                        if im_height > 300:
                            box_temp.append([int(row["xmin"]-x),int(row["ymin"]),int(row["xmax"]-x),int(300)])
                            label_tem.append(1)
                            n_box += 1
                        else:
                            box_temp.append(
                                [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(row["ymax"])])
                            label_tem.append(1)
                            n_box += 1
                objects.append({"boxes": box_temp, "labels": label_tem})
                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                if im_height < 300:
                    padding_amount = 300 - im_height
                    im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    cv2.imwrite(dir_save, im_pad)
                    images.append(dir_save)
                    total_image += 1
                else:
                    cv2.imwrite(dir_save, crop_im)
                    images.append(dir_save)
                    total_image += 1

                x += 300

            if im_height < 300:
                box_temp = []
                label_tem = []
                for index, row in bbox_df.iterrows():
                    if row["xmin"] > im_width-300:
                        box_temp.append([int(row["xmin"]-(im_width-300)),int(row["ymin"]),int(row["xmax"]-(im_width-300)),int(row["ymax"])])
                        label_tem.append(1)
                        n_box += 1

                objects.append({"boxes": box_temp, "labels": label_tem})

                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                x, y = im_width - 300, 0
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                padding_amount = 300 - im_height
                im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                cv2.imwrite(dir_save, im_pad)
                images.append(dir_save)
                total_image += 1
            else:
                box_temp = []
                label_tem = []
                for index, row in bbox_df.iterrows():
                    if row["xmin"] > im_width-300:
                        box_temp.append([int(row["xmin"]-(im_width-300)),int(row["ymin"]),int(row["xmax"]-(im_width-300)),int(300)])
                        label_tem.append(1)
                        n_box += 1
                objects.append({"boxes": box_temp, "labels": label_tem})

                j = j.replace('.tif','_')
                name = j + str(total_image) + ".tif"
                dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                x, y = im_width - 300, 0
                width, height = 300, 300
                crop_im = im[y:y + height, x:x + width]
                cv2.imwrite(dir_save, crop_im)
                images.append(dir_save)
                total_image += 1
    with open(os.path.join(root_dir, 'Images.json'), 'w') as j:
        json.dump(images, j)

    with open(os.path.join(root_dir, 'Bboxes.json'), 'w') as j:
        json.dump(objects, j)
    print('\nCropped original images and created %d train and test images. Images has been saved to %s .' % (total_image-1,root_dir))
    print('\nFound %d bounding boxes. Image path and bbox locations has ben saved to %s' % (n_box,root_dir))

#create_images_and_bbox_list("/home/criuser/Desktop/Internship/Orginal_images", '/home/criuser/Desktop/Internship/Orginal_measure','/home/criuser/Desktop/Internship/Output')
def display_image(dir_json_images,dir_json_bbx,n_display):
    with open(dir_json_images) as f:
        dir_images = json.load(f)
    with open(dir_json_bbx) as f:
        boxes = json.load(f)

    count = 0
    for i,b in zip(dir_images, boxes):
        img = cv2.imread(i)
        bboxes = b["boxes"]

        for j in bboxes:
            img = cv2.rectangle(img, (j[0], j[1]), (j[2], j[3]), (255, 0, 0), 1)
        count += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        if count == n_display:
            break


display_image('/home/criuser/Desktop/Internship/Output/Images.json','/home/criuser/Desktop/Internship/Output/Bboxes.json',2)




