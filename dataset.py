import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class EcoliBacteriaDataset(Dataset):

    def __init__(self,data_folder,split):
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        with open(os.path.join(data_folder, self.split + 'Images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + 'Objects.json'), 'r') as j:
            self.objects = json.load(j)