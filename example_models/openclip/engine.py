import os
from vl_checklist.vlp_model import VLPModel
from example_models.utils.helpers import LRUCache, chunks
import torch.cuda
import clip
from PIL import Image
import numpy as np
import open_clip

class OPEN_CLIP(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    MAX_CACHE = 20

    def __init__(self,model_id, model_path):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "resources"
        self.model_id = model_id
        self.model_path = model_path

        self.model, self.preprocess = self._load_model(model_id, model_path)

    def model_name(self):
        return self.model_id

    def _load_model(self, model_id, model_path):
        if model_id is None:
            raise Exception("Model ID cannot be None.")
    
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_id, pretrained=model_path, device=self.device)
        return model.eval(), preprocess_val

    def _load_data(self, src_type, data):
        pass

    def predict(self,
                images: list,
                texts: list,
                src_type: str = 'local'
                ):
        images_batch = images.to(self.device)
        texts_batch = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images_batch, texts_batch)
        
        similarity = image_features @ text_features.T * logit_scale

        return torch.diagonal(similarity).cpu().numpy().astype(np.float64)

