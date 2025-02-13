from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import yaml
import json
import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import pickle
import pycocotools.mask as mask_util
from torchvision.transforms import Normalize


# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, corpus_name, type, dataset_dir='', task='itm', version="v1", transform=None, objects_sense_format=None, objects_data=None) -> None:
        self.root_dir = os.path.dirname(os.path.realpath(__file__))#.replace("/vl_checklist", "")
        self.cur_dir = os.path.realpath(os.curdir)
        self.dataset_dir = dataset_dir
        self.version = version
        self.transform = transform
        self.objects_transform = copy.deepcopy(transform)
        self.objects_transform.transforms = self.objects_transform.transforms[:2]
        self.objects_sense_normalize = Normalize(mean=[0.5], std=[0.26])
        self.objects_sense_format = objects_sense_format

        self.config = yaml.load(open(os.path.join(self.cur_dir, 'corpus',  self.version, type,f'{corpus_name}.yaml'), 'r'), Loader=yaml.FullLoader)
        self.data = json.load(open(os.path.join(self.cur_dir, self.config["ANNO_PATH"])))
        self.objects_data = objects_data

    def __len__(self):
        return len(self.data)

    def load_edges(self, edges_demo_path, image_shape):
        if os.path.exists(edges_demo_path):
            with open(edges_demo_path, 'rb') as f:
                combined_edges = pickle.load(f)
            rle = {'size': combined_edges['size'], 'counts': combined_edges['counts']}
            mask = mask_util.decode(rle)
            return mask
        else:
            print(f"Edges file not found: {edges_demo_path}")
            return np.ones(image_shape[:2], dtype=np.uint8)

    def get_objects_sense(self, filename, image, objects_sense_format, objects_data):
        if objects_sense_format == 'edges':
            key, _ = os.path.splitext(filename)
            objects_sense_path = os.path.join(objects_data, self.config["IMG_ROOT"], key + '_edges.pkl')
            edges = self.load_edges(objects_sense_path, image.size)

        edges = torch.as_tensor(edges).unsqueeze(0).float() * 255
        edges = self.objects_sense_normalize(edges)
        return edges

    def __getitem__(self, idx):
        file_name, texts_dict = self.data[idx]
        path = os.path.join(self.dataset_dir, self.config["IMG_ROOT"], file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        objects_sense = ''
        img = Image.open(path)

        if self.objects_sense_format:
            # !!!!!!!! Processing the file_name to object key !!!!!!!!!!
            objects_sense = self.get_objects_sense(file_name, img, self.objects_sense_format, self.objects_data)
            objects_sense = self.objects_transform(objects_sense)

        img = self.transform(img)

        return {
            "img": img,
            "path": path,
            "texts_pos": texts_dict["POS"][0],
            "texts_neg": texts_dict["NEG"][0],
            "objects_sense": objects_sense
        }

