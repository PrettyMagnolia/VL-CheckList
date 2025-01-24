from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os
import yaml
import json
from torch.utils.data import Dataset


# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, corpus_name, type, dataset_dir='', task='itm', version="v1", transform=None) -> None:
        self.root_dir = os.path.dirname(os.path.realpath(__file__))#.replace("/vl_checklist", "")
        self.cur_dir = os.path.realpath(os.curdir)
        self.dataset_dir = dataset_dir
        self.version = version
        self.transform = transform

        self.config = yaml.load(open(os.path.join(self.cur_dir, 'corpus',  self.version, type,f'{corpus_name}.yaml'), 'r'), Loader=yaml.FullLoader)
        self.data = json.load(open(os.path.join(self.cur_dir, self.config["ANNO_PATH"])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, texts_dict = self.data[idx]
        path = os.path.join(self.dataset_dir, self.config["IMG_ROOT"], path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
    
        return {
            "img": self.transform(Image.open(path)),
            "path": path,
            "texts_pos": texts_dict["POS"][0],
            "texts_neg": texts_dict["NEG"][0]
        }

