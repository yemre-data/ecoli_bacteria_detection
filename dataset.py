import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

class EcoliBacteriaDataset(Dataset):

    def __init__(self,data_folder,train_test):
        self.train_test = train_test.upper()

        assert self.train_test in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        with open(os.path.join(data_folder, self.train_test + '_IMAGES.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.train_test + '_BBOXES.json'), 'r') as j:
            self.bboxes = json.load(j)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        bboxes = self.bboxes[i]
        boxes = torch.FloatTensor(bboxes['boxes'])
        labels = torch.LongTensor(bboxes['labels'])
        image, boxes, labels = transform(image, boxes, labels, split = self.train_test)
        return image, boxes, labels

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    images = list()
    boxes = list()
    labels = list()
    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)

    return images, boxes, labels