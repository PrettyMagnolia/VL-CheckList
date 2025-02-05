from example_models.openclip.engine import OPEN_CLIP
from vl_checklist.evaluate import Evaluate


if __name__ == '__main__':
    open_clip_vit_b_16 = OPEN_CLIP(model_id='ViT-B-16', model_path='/mnt/shared/unibench/models/open-clip/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin')
    clip_eval = Evaluate("configs/test_all.yaml", model=open_clip_vit_b_16)
    clip_eval.start()
    


