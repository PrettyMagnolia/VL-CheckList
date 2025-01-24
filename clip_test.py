from example_models.clip.engine import CLIP
from vl_checklist.evaluate import Evaluate


if __name__ == '__main__':
    clip_vit_b_16 = CLIP('/mnt/shared/unibench/models/clip/ViT-B-16.pt')
    clip_eval = Evaluate("configs/test_all.yaml", model=clip_vit_b_16)
    clip_eval.start()
    


