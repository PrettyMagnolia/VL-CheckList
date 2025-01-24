import os
from vl_checklist.vlp_model import VLPModel
from example_models.utils.helpers import LRUCache, chunks
import torch.cuda
import clip
from PIL import Image
import numpy as np

class CLIP(VLPModel):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    MAX_CACHE = 20

    def __init__(self,model_id):
        self._models = LRUCache(self.MAX_CACHE)
        self.batch_size = 16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "resources"
        self.model_id = model_id

        model_list = self._load_model(model_id)
        self.model = model_list[0].eval()
        self.preprocess = model_list[1]

    def model_name(self):
        return self.model_id

    def _load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")
    
        if not self._models.has(model_id):
            model, preprocess = clip.load(model_id, device=self.device)
            self._models.put(model_id, [model, preprocess])
        return self._models.get(model_id)

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
            logits_per_image, logits_per_text = self.model(images, texts_batch)

        return torch.diagonal(logits_per_image).cpu().numpy().astype(np.float64)

