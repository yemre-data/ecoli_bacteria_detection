from operator import ne
import pandas as pd
import cv2
import numpy as np
import torch
import json
import os, shutil
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage import io
import torchvision.transforms.functional as FT
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_images_and_bbox_list(dir_image_folder, dir_csv_folder, dir_save):
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

    for j, b in zip(image_files, bbox_files):
        dir_image = os.path.join(os.path.expanduser('~'), dir_image_folder, j)
        dir_bbox = os.path.join(os.path.expanduser('~'), dir_csv_folder, b)

        im_init = io.imread(dir_image)
        im_init = np.split(im_init, im_init.shape[0], 0)
        for slice_num in range(0, len(im_init)):
            im = np.reshape(im_init[slice_num],
                            (im_init[slice_num].shape[1], im_init[slice_num].shape[2], im_init[slice_num].shape[3]))
            im_height, im_width = im.shape[0], im.shape[1]
            bbox_df = pd.read_csv(dir_bbox)
            bbox_df = bbox_df.loc[bbox_df['Slice'] == slice_num + 1]
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
                    for index, row in bbox_df.iterrows():
                        if x <= row["xmin"] <= x + 300 and row["xmin"] < nm_total_image * 300:
                            if im_height > 300:
                                box_temp.append(
                                    [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(300)])
                                label_tem.append(1)
                                n_box += 1
                            else:
                                box_temp.append(
                                    [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(row["ymax"])])
                                label_tem.append(1)
                                n_box += 1
                    if box_temp != [] and label_tem != []:

                        objects.append({"boxes": box_temp, "labels": label_tem})

                        j = j.replace('.tif', '_')
                        name = j + str(total_image) + ".jpg"
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                        width, height = 300, 300
                        crop_im = im[y:y + height, x:x + width]
                        if im_height < 300:
                            padding_amount = 300 - im_height
                            im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT,
                                                        (0, 0, 0))
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
                        if x <= row["xmin"] <= x + 300 and row["xmin"] < nm_total_image * 300:
                            if im_height > 300:
                                box_temp.append(
                                    [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(300)])
                                label_tem.append(1)
                                n_box += 1
                            else:
                                box_temp.append(
                                    [int(row["xmin"] - x), int(row["ymin"]), int(row["xmax"] - x), int(row["ymax"])])
                                label_tem.append(1)
                                n_box += 1
                    if box_temp != [] and label_tem != []:
                        objects.append({"boxes": box_temp, "labels": label_tem})
                        j = j.replace('.tif', '_')
                        name = j + str(total_image) + ".jpg"
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                        width, height = 300, 300
                        crop_im = im[y:y + height, x:x + width]
                        if im_height < 300:
                            padding_amount = 300 - im_height
                            im_pad = cv2.copyMakeBorder(crop_im, 0, padding_amount, 0, 0, cv2.BORDER_CONSTANT,
                                                        (0, 0, 0))
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
                        if row["xmin"] > im_width - 300:
                            box_temp.append([int(row["xmin"] - (im_width - 300)), int(row["ymin"]),
                                             int(row["xmax"] - (im_width - 300)), int(row["ymax"])])
                            label_tem.append(1)
                            n_box += 1
                    if box_temp != [] and label_tem != []:
                        objects.append({"boxes": box_temp, "labels": label_tem})

                        j = j.replace('.tif', '_')
                        name = j + str(total_image) + ".jpg"
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
                        if row["xmin"] > im_width - 300:
                            box_temp.append([int(row["xmin"] - (im_width - 300)), int(row["ymin"]),
                                             int(row["xmax"] - (im_width - 300)), int(300)])
                            label_tem.append(1)
                            n_box += 1
                    if box_temp != [] and label_tem != []:
                        objects.append({"boxes": box_temp, "labels": label_tem})

                        j = j.replace('.tif', '_')
                        name = j + str(total_image) + ".jpg"
                        dir_save = os.path.join(os.path.expanduser('~'), root_dir, name)
                        x, y = im_width - 300, 0
                        width, height = 300, 300
                        crop_im = im[y:y + height, x:x + width]
                        cv2.imwrite(dir_save, crop_im)
                        images.append(dir_save)
                        total_image += 1

    with open(os.path.join(root_dir, 'ALL' + '_IMAGES.json'), 'w') as j:
        json.dump(images, j)

    with open(os.path.join(root_dir, 'ALL' + '_BBOXES.json'), 'w') as j:
        json.dump(objects, j)
    # Train and test split randomly
    test_images_ind = random.sample(range(1, len(images)), int(0.1 * len(images)) + 1)
    test_images = []
    test_objects = []
    train_images = images.copy()
    train_objects = objects.copy()
    for i in test_images_ind:
        test_images.append(images[i])
        train_images.remove(images[i])
        test_objects.append(objects[i])
        train_objects.remove(objects[i])

    with open(os.path.join(root_dir, 'TRAIN' + '_IMAGES.json'), 'w') as j:
        json.dump(train_images, j)

    with open(os.path.join(root_dir, 'TRAIN' + '_BBOXES.json'), 'w') as j:
        json.dump(train_objects, j)

    with open(os.path.join(root_dir, 'TEST' + '_IMAGES.json'), 'w') as j:
        json.dump(test_images, j)

    with open(os.path.join(root_dir, 'TEST' + '_BBOXES.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nCropped original images and created %d train and %d test images. Images has been saved to %s .' % (
    len(train_images), len(test_images), root_dir))
    print('\nFound %d bounding boxes as train and test. Image path and bbox locations has ben saved to %s. ' % (
    n_box, root_dir))


# create_images_and_bbox_list("/home/criuser/Desktop/Internship/Orginal_images", '/home/criuser/Desktop/Internship/Orginal_measure','/home/criuser/Desktop/Internship/Output')

def display_image(dir_json_images, dir_json_bbx, n_display):
    with open(dir_json_images) as f:
        dir_images = json.load(f)
    with open(dir_json_bbx) as f:
        boxes = json.load(f)

    count = 0
    for i, b in zip(dir_images, boxes):
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


# display_image('/home/criuser/Desktop/Internship/Output/TRAIN_IMAGES.json','/home/criuser/Desktop/Internship/Output/TRAIN_BBOXES.json',10)
def random_crop(image, boxes, labels):

    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

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

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):

    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def photometric_distort(image):

    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255     because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):

    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, boxes, labels, split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        # new_image = photometric_distort(new_image)
        # new_image = FT.to_pil_image(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)
        # Convert PIL image to Torch tensor
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels


def decimate(tensor, m):

    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def xy_to_cxcy(xy):

    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):

    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):


    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):


    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):


    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):


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


def save_checkpoint(epoch, model, optimizer, file_name):
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