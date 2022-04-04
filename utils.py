
import pandas as pd
import cv2
import numpy as np
import json
import os, shutil
from os.path import isfile, join
from skimage import io
import torchvision.transforms.functional as FT
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
voc_labels = ('bacteria')
label_map = {'bacteria': 1}
label_map['background'] = 0
rev_label_map = {1:'bacteria',0:'background'}  # Inverse mapping

# Color map for bounding box
distinct_colors = ['#e6194b', '#3cb44b']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

'''
This function is creating image(300*300) data and bbox location data from original images and initially we added 
difficulties info as no-difficulties.
param:  dir_image_folder : original images path should include only images
        dir_csv_folder : csv folder belonging to images that has been gotten from IMAGEJ program.
        dir_save : path of cropped image images and bbox.
'''

def create_images_and_bbox_list(dir_image_folder,dir_csv_folder, dir_save):
    """
    This function is creating image(300*300) data and bbox location data from original images and initially we added
    difficulties info as no-difficulties.
    :param dir_image_folder: original images path should include only images
    :param dir_csv_folder: csv folder belonging to images that has been gotten from IMAGEJ program
    :param dir_save: path of cropped image images and bbox
    :return:
    """
    # getting image files
    image_files = [f for f in os.listdir(dir_image_folder) if isfile(join(dir_image_folder, f))]
    #getting bbox files
    bbox_files = [f for f in os.listdir(dir_csv_folder) if isfile(join(dir_csv_folder, f))]
    #sort based on the name
    image_files = sorted(image_files)
    # sort based on the name
    bbox_files = sorted(bbox_files)
    #will use for naming
    total_image = 1
    #copy save path
    root_dir_save = dir_save
    #list of images
    images = list()
    #list 0f objects with bbox and dificulties
    objects = list()
    #number of bbox
    n_box = 0
    # deleting save path before save new files be careful, when you are running you should make copy your valuable files
    for filename in os.listdir(dir_save):
        file_path = os.path.join(dir_save, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass
    # here unstack images and get one by one all images and cropped them as 300*300 and getting corresponding bboxes
    # if our height of original image is less than 300 we are filling as black to reach appropriate resolution for model

    # loop for get image and bbox_files
    for j,b in zip(image_files,bbox_files):
        #path join
        dir_image = os.path.join(os.path.expanduser('~'), dir_image_folder, j)
        dir_bbox = os.path.join(os.path.expanduser('~'), dir_csv_folder, b)
        #img read
        im_init = io.imread(dir_image)
        # stacked image unstack
        im_init = np.split(im_init, im_init.shape[0], 0)
        # loop for iterate unstacked image one by one
        for slice_num in range(0,len(im_init)):
            # reshape image array
            im = np.reshape(im_init[slice_num], (im_init[slice_num].shape[1], im_init[slice_num].shape[2], im_init[slice_num].shape[3]))
            # get im_height and im_width
            im_height, im_width = im.shape[0], im.shape[1]
            # read bbox csv file
            bbox_df = pd.read_csv(dir_bbox)
            # get corresponded bbox
            bbox_df = bbox_df.loc[bbox_df['Slice'] == slice_num+1]
            # filter dataframe
            bbox_df = bbox_df[['BX', 'BY', 'Width', 'Height']]
            # create new dict for have exact bbox dimension
            bbox_df = {"xmin": bbox_df['BX'], "ymin": bbox_df['BY'], "xmax": bbox_df['BX'] + bbox_df['Width'],
                      "ymax": bbox_df['BY'] + bbox_df['Height']}
            bbox_df = pd.DataFrame(bbox_df)
            # check it dived by 300 or not
            if im_width % 300 == 0:
                # get number of total cropped-image
                nm_total_image = im_width / 300
                nm_total_image = int(nm_total_image)
                # define initial location
                x, y = 0, 0
                # cropping loop
                for i in range(0, nm_total_image):
                    # list of bbox,label,difficulties temporary
                    box_temp = []
                    label_tem = []
                    dif_tem = []
                    # loop for getting each belonging bbox
                    for index,row in bbox_df.iterrows():
                        # condition width of cropped image is in between x+300 and unstacked image width
                        if x <= row["xmin"] <= x+300 and row["xmin"] < nm_total_image*300:
                            # check height if greater than 300 than add y_max as 300
                            if im_height > 300:
                                # add to box_tep list bbox location
                                box_temp.append([int(row["xmin"]-x),int(row["ymin"]),int(row["xmax"]-x),int(300)])
                                # label are the same
                                label_tem.append(1)
                                # now added difficulties as 0 but we can change looking each anotated bacteria to define
                                # their difficulties.
                                dif_tem.append(0)
                                # count bboxes
                                n_box += 1
                            # height is less than 300 get location as original y_max
                            else:
                                # same appends
                                box_temp.append(
                                    [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(row["ymax"])])
                                label_tem.append(1)
                                dif_tem.append(0)
                                n_box += 1
                    # check our box and label empty or not
                    if box_temp != [] and label_tem !=[]:
                        # add each bacteria info
                        objects.append({"boxes":box_temp,"labels":label_tem,"difficulties":dif_tem})
                        # changing image type
                        j = j.replace('.tif','_')
                        name = j + str(total_image) + ".jpg"
                        # create save location with image name
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir_save, name)
                        width, height = 300, 300
                        # cropped image
                        crop_im = im[y:y + height, x:x + width]
                        # check if cropped image height less than 300
                        if im_height < 300:
                            # padding amount
                            padding_amount = 300 - im_height
                            # padding by black color
                            im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                            # saving
                            cv2.imwrite(dir_save, im_pad)
                            # append location of image
                            images.append(dir_save)
                            # count image
                            total_image += 1
                        else:
                            # save image
                            cv2.imwrite(dir_save, crop_im)
                            # add dir of images
                            images.append(dir_save)
                            # counting total_image
                            total_image += 1
                    # increasing width of the image every iter
                    x += 300
            # if our unstacked image of the width is not divisible by 300 then get last 300 part again to not lose data.
            else:
                # get number of total image
                nm_total_image = im_width / 300
                # add one more
                nm_total_image = int(nm_total_image) + 1
                # define initial location
                x, y = 0, 0
                # cropping loop
                for i in range(1, nm_total_image):
                    # list of bbox,label,difficulties temporary
                    box_temp = []
                    label_tem = []
                    dif_tem = []
                    # loop for getting each belonging bbox
                    for index, row in bbox_df.iterrows():
                        # condition width of cropped image is in between x+300 and unstacked image width
                        if x <= row["xmin"] <= x+300 and row["xmin"] < nm_total_image*300:
                            # check height if greater than 300 than add y_max as 300
                            if im_height > 300:
                                # add to box_tep list bbox location
                                box_temp.append([int(row["xmin"]-x),int(row["ymin"]),int(row["xmax"]-x),int(300)])
                                # label are the same
                                label_tem.append(1)
                                # now added difficulties as 0 but we can change looking each annotated bacteria to define
                                # their difficulties.
                                dif_tem.append(0)
                                # counting bboxex
                                n_box += 1
                            else:
                                # same appends
                                box_temp.append(
                                    [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(row["ymax"])])
                                label_tem.append(1)
                                dif_tem.append(0)
                                n_box += 1
                    # add each bacteria info
                    if box_temp != [] and label_tem != []:
                        # add each bacteria info
                        objects.append({"boxes": box_temp, "labels": label_tem,"difficulties":dif_tem})
                        # changing image type
                        j = j.replace('.tif','_')
                        name = j + str(total_image) + ".jpg"
                        #  create save location with image name
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir_save, name)
                        width, height = 300, 300
                        # cropped image
                        crop_im = im[y:y + height, x:x + width]
                        # check if cropped image heights less than 300
                        if im_height < 300:
                            padding_amount = 300 - im_height
                            # padding by black color
                            im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                            # save cropped image
                            cv2.imwrite(dir_save, im_pad)
                            # add dir of image
                            images.append(dir_save)
                            # counting total image
                            total_image += 1
                        else:
                            # save image
                            cv2.imwrite(dir_save, crop_im)
                            # add dir of image
                            images.append(dir_save)
                            # count total image
                            total_image += 1
                    # increasing width of the image every iter
                    x += 300
                # check if unstacked image heights less than 300
                if im_height < 300:
                    # list of bbox,label,difficulties temporary
                    box_temp = []
                    label_tem = []
                    dif_tem = []
                    # loop for getting each belonging bbox
                    for index, row in bbox_df.iterrows():

                        if row["xmin"] > im_width-300:
                            box_temp.append([int(row["xmin"]-(im_width-300)),int(row["ymin"]),int(row["xmax"]-(im_width-300)),int(row["ymax"])])
                            label_tem.append(1)
                            dif_tem.append(0)
                            n_box += 1
                    if box_temp != [] and label_tem != []:
                        objects.append({"boxes": box_temp, "labels": label_tem,"difficulties":dif_tem})

                        j = j.replace('.tif','_')
                        name = j + str(total_image) + ".jpg"
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir_save, name)
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
                    dif_tem = []
                    for index, row in bbox_df.iterrows():
                        if row["xmin"] > im_width-300:
                            box_temp.append([int(row["xmin"]-(im_width-300)),int(row["ymin"]),int(row["xmax"]-(im_width-300)),int(300)])
                            label_tem.append(1)
                            dif_tem.append(0)
                            n_box += 1
                    # check our box and label empty or not
                    if box_temp != [] and label_tem != []:

                        objects.append({"boxes": box_temp, "labels": label_tem,"difficulties":dif_tem})

                        j = j.replace('.tif','_')
                        name = j + str(total_image) + ".jpg"
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir_save, name)
                        x, y = im_width - 300, 0
                        width, height = 300, 300
                        crop_im = im[y:y + height, x:x + width]
                        cv2.imwrite(dir_save, crop_im)
                        images.append(dir_save)
                        total_image += 1

    with open(os.path.join(root_dir_save, 'ALL' + '_IMAGES.json'), 'w') as j:
        json.dump(images, j)

    with open(os.path.join(root_dir_save, 'ALL' + '_BBOXES.json'), 'w') as j:
        json.dump(objects, j)

    # Train and test split randomly
    test_images_ind = random.sample(range(1, len(images)), int(0.1 * len(images))+1)
    test_images = []
    test_objects = []
    train_images = images.copy()
    train_objects = objects.copy()
    for i in test_images_ind:
        test_images.append(images[i])
        train_images.remove(images[i])
        test_objects.append(objects[i])
        train_objects.remove(objects[i])

    #saving files into the save_dir
    with open(os.path.join(root_dir_save, 'TRAIN' + '_IMAGES.json'), 'w') as j:
        json.dump(train_images, j)

    with open(os.path.join(root_dir_save, 'TRAIN' + '_BBOXES.json'), 'w') as j:
        json.dump(train_objects, j)

    with open(os.path.join(root_dir_save, 'TEST' + '_IMAGES.json'), 'w') as j:
        json.dump(test_images, j)

    with open(os.path.join(root_dir_save, 'TEST' + '_BBOXES.json'), 'w') as j:
        json.dump(test_objects, j)


    print('\nCropped original images and created %d train and %d test images. Images has been saved to %s .' % (len(train_images), len(test_images), root_dir_save))
    print('\nFound %d bounding boxes as train and test. Image path and bbox locations has ben saved to %s. ' % (n_box, root_dir_save))


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    print(label_map)
    print(rev_label_map)
    print()
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def fix_boxes(image_boxes):

    """
    This function is fixing final bounding boxes. Our widths are same,so we can fix them as 24 pixel.
    0.08*300 = 24 pixels
    :param image_boxes: final image boxes
    :return: fixed image_boxes
    """

    for i in range(len(image_boxes)):
        x1 = image_boxes[i][0].item()
        x1 = x1 * 300
        if x1 < 5:
            image_boxes[i][0] = 0.0
            image_boxes[i][2] = 0.08
        elif 5 < x1 < 35:
            image_boxes[i][0] = 0.1
            image_boxes[i][2] = 0.18
        elif 35 < x1 < 65:
            image_boxes[i][0] = 0.2
            image_boxes[i][2] = 0.28
        elif 65 < x1 < 95:
            image_boxes[i][0] = 0.3
            image_boxes[i][2] = 0.38
        elif 95 < x1 < 125:
            image_boxes[i][0] = 0.4
            image_boxes[i][2] = 0.48
        elif 125 < x1 < 155:
            image_boxes[i][0] = 0.5
            image_boxes[i][2] = 0.58
        elif 155 < x1 < 185:
            image_boxes[i][0] = 0.6
            image_boxes[i][2] = 0.68
        elif 185 < x1 < 215:
            image_boxes[i][0] = 0.7
            image_boxes[i][2] = 0.78
        elif 215 < x1 < 245:
            image_boxes[i][0] = 0.8
            image_boxes[i][2] = 0.88
        elif 245 < x1 < 275:
            image_boxes[i][0] = 0.9
            image_boxes[i][2] = 0.98

    return image_boxes


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    Here we change also center-width of the bounding boxes as 0.0446. It will match with priors center-widths.
    With this way we are training our model with pre-define width bounding box.
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """

    pre_define = torch.cuda.FloatTensor([0.0446])
    pre_define = pre_define.repeat(8732, 1)

    pre_define = torch.cat([pre_define, xy[:, 3:] - xy[:, 1:2]], 1)

    cxcy = torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, pre_define], 1)

    return cxcy


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """

    xy = torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                    cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

    return xy


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155

    gcxgcy = torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
                        torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h

    return gcxgcy


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = '/content/drive/MyDrive/Bacteria/CheckPoints/checkpoint_predefined_7_lr03_32.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



