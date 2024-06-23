from concurrent.futures import ThreadPoolExecutor
from cv2 import imread, imwrite
from glob import glob
from models.automatic_mask_generator import SamAutomaticMaskGenerator
from numpy import int16, savez_compressed
from os import environ, makedirs
from os.path import dirname, basename, join
from segment_anything import build_sam_vit_h
import torch
from tqdm import tqdm

import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--ckpt_path', type=str, default="/data/haiyang/projects/segment-anything/ckpts/sam_vit_h_4b8939.pth",
                        help='SAM checkpoint file')
    parser.add_argument('--file_path', type=str, default="/data/haiyang/projects/OmniSeg3D_release/data/llff_flower/images_4",
                        help='RGB image folder path')
    
    return parser.parse_args()


@torch.inference_mode()
def process_images(ckpt_path: str, files: list):
    sam_generator = SamAutomaticMaskGenerator(
        build_sam_vit_h(ckpt_path).requires_grad_(False).cuda(),
        points_per_side=32,
        points_per_batch=64,      # 256
        pred_iou_thresh=.88,
        stability_score_thresh=.9,  # default: 0.95, LLFF: 0.9
        stability_score_offset=1,
        box_nms_thresh=.7,
        crop_n_layers=1,  # default: 0, LLFF: 1
        crop_nms_thresh=.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=128
    )
    for file in tqdm(files, leave=False):
        prefixes = ('masks', 'patches', 'sam')
        paths = [join(dirname(dirname(file)), p, basename(file)) for p in prefixes]
        [makedirs(dirname(p), exist_ok=True) for p in paths]
        
        # --- Generate segmentation masks --- #
        masks = sam_generator.generate(image := imread(file))
        masks = masks[masks.sum((1, 2)).argsort()]
        
        # --- Refine and visualize segmentation masks --- #
        vis = torch.zeros_like(image := torch.tensor(image) >> 2)
        for m, mask in enumerate(masks):
            union = (mask & masks[m + 1:]).sum((1, 2), True)
            # masks[m + 1:] |= mask & (union > .9 * mask.sum(None, True))
            masks[m + 1:] |= mask & (union > .9 * mask.sum((0, 1), True))
            vis[mask] = torch.randint(192, (3,), dtype=torch.uint8)
        imwrite(paths[0], vis.add_(image).cpu().numpy())
        
        # --- Identify and visualize disjoint patches --- #
        unique, indices = masks.flatten(1).unique(return_inverse=True, dim=1)
        (cm := torch.randint(192, (unique.size(1), 3), dtype=torch.uint8))[0] = 0
        imwrite(paths[1], cm[indices.cpu()].view_as(image).add_(image).cpu().numpy())
        
        # --- Calculate patch correlation --- #
        unique, indices = unique.half(), indices.view_as(mask).cpu().numpy()
        corr = (unique.T @ unique).byte().cpu().numpy()
        savez_compressed(paths[2][:-4], indices=int16(indices), correlation=corr)

        torch.cuda.empty_cache()


if __name__ == '__main__':

    hparams = get_opts()
    
    ckpt_path = hparams.ckpt_path
    file_path = hparams.file_path
    
    files = sorted(glob(join(file_path, "*.*")))

    process_images(ckpt_path, files)
