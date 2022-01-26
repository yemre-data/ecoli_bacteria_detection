
import pandas as pd
import cv2
import numpy as np
import torch
import json
import os,shutil
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage import io
import torchvision.transforms.functional as FT
import torch
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        im_init = io.imread(dir_image)
        im_init = np.split(im_init, im_init.shape[0], 0)
        for slice_num in range(0,len(im_init)):
            im = np.reshape(im_init[slice_num], (im_init[slice_num].shape[1], im_init[slice_num].shape[2], im_init[slice_num].shape[0]))
            im_height, im_width = im.shape[0], im.shape[1]
            bbox_df = pd.read_csv(dir_bbox)
            bbox_df = bbox_df.loc[bbox_df['Slice'] == slice_num+1]
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
                    if box_temp != [] and label_tem !=[]:

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
                    if box_temp != [] and label_tem != []:
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
                    if box_temp != [] and label_tem != []:
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
                    if box_temp != [] and label_tem != []:
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

    with open(os.path.join(root_dir, 'ALL' +'_IMAGES.json'), 'w') as j:
        json.dump(images, j)

    with open(os.path.join(root_dir, 'ALL' +'_BBOXES.json'), 'w') as j:
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


    with open(os.path.join(root_dir, 'TRAIN' +'_IMAGES.json'), 'w') as j:
        json.dump(train_images, j)

    with open(os.path.join(root_dir, 'TRAIN' +'_BBOXES.json'), 'w') as j:
        json.dump(train_objects, j)

    with open(os.path.join(root_dir, 'TEST' +'_IMAGES.json'), 'w') as j:
        json.dump(test_images, j)

    with open(os.path.join(root_dir, 'TEST' +'_BBOXES.json'), 'w') as j:
        json.dump(test_objects, j)


    print('\nCropped original images and created %d train and %d test images. Images has been saved to %s .' % (total_image-1-int(0.1*len(images)), int(0.1*len(images)),root_dir))
    print('\nFound %d bounding boxes as train and test. Image path and bbox locations has ben saved to %s. ' % (n_box,root_dir))

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

#display_image('/home/criuser/Desktop/Internship/Output/TRAIN_IMAGES.json','/home/criuser/Desktop/Internship/Output/TRAIN_BBOXES.json',10)

def transform(image, boxes, labels, split):

    assert split in {'TRAIN', 'TEST'}
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    new_image = image
    new_boxes = boxes
    new_labels = labels

    if split == 'TRAIN':

        new_image = FT.to_tensor(new_image)

    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels
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

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


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
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


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



def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

def save_checkpoint(epoch, model, optimizer,file_name):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = file_name
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