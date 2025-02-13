import argparse
from example_models.openclip.engine import OPEN_CLIP
from vl_checklist.evaluate import Evaluate

def main(args):
    open_clip = OPEN_CLIP(model_id=args.model_id, model_path=args.model_path)
    clip_eval = Evaluate(config_file=args.config_path, model=open_clip, objects_sense_format=args.objects_sense_format, objects_data=args.objects_data)
    clip_eval.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OPEN_CLIP evaluation")

    parser.add_argument("--model_id", type=str, default="ViT-B-16",
                        help="Model ID for OPEN_CLIP")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the OPEN_CLIP model checkpoint")
    parser.add_argument("--config_path", type=str, default="configs/test_all.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--objects_sense_format", type=str, default=None,
                        help="Format of objects sense")
    parser.add_argument("--objects_data", type=str, default=None,
                        help="Path to file(s) with objects data")

    args = parser.parse_args()

    import time
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    start_time = time.time()

    main(args)
    
    print(f"Execution time: {time.time() - start_time}")

