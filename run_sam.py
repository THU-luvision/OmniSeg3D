from concurrent.futures import ThreadPoolExecutor
from cv2 import imread, imwrite
from glob import glob
from models.automatic_mask_generator import SamAutomaticMaskGenerator
from numpy import int16, savez_compressed
from os import environ, makedirs
from os.path import dirname
from segment_anything import build_sam_vit_h
import torch
from tqdm import tqdm


@torch.inference_mode()
def process_images(rank: int, files: list):
    torch.cuda.set_device(rank)
    sam_generator = SamAutomaticMaskGenerator(
        # build_sam_vit_h('sam_vit_h_4b8939.pth').requires_grad_(False),
        build_sam_vit_h("/media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/Segment/SegmentAnythingin3D/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth").requires_grad_(False).to(rank),
        points_per_side=32,
        points_per_batch=32,      # 256
        pred_iou_thresh=.88,
        stability_score_thresh=.90,  # .9 for LLFF
        stability_score_offset=1,
        box_nms_thresh=.7,
        crop_n_layers=0,  # 1 for LLFF
        crop_nms_thresh=.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=128
    )
    for file in tqdm(files, leave=False, position=rank):
        prefixes = ('rgb', 'masks', 'patches', 'sam')
        # prefixes = ('rgb', 'masks', 'patches', 'sam') if 'Replica' in file \
        #     else ('images', 'masks', 'patches', 'sam')
        paths = [file.replace(prefixes[0], p) for p in prefixes[1:]]
        [makedirs(dirname(p), exist_ok=True) for p in paths]
        # Generate segmentation masks
        masks = sam_generator.generate(image := imread(file))
        masks = masks[masks.sum((1, 2)).argsort()]
        # Refine and visualize segmentation masks
        vis = torch.zeros_like(image := torch.tensor(image) >> 2)
        for m, mask in enumerate(masks):
            union = (mask & masks[m + 1:]).sum((1, 2), True)
            # masks[m + 1:] |= mask & (union > .9 * mask.sum(None, True))
            masks[m + 1:] |= mask & (union > .9 * mask.sum((0, 1), True))
            vis[mask] = torch.randint(192, (3,), dtype=torch.uint8)
        imwrite(paths[0], vis.add_(image).cpu().numpy())
        # Identify and visualize disjoint patches
        unique, indices = masks.flatten(1).unique(return_inverse=True, dim=1)
        (cm := torch.randint(192, (unique.size(1), 3), dtype=torch.uint8))[0] = 0
        imwrite(paths[1], cm[indices].view_as(image).add_(image).cpu().numpy())
        # Calculate patch correlation
        unique, indices = unique.half(), indices.view_as(mask).cpu().numpy()
        corr = (unique.T @ unique).byte().cpu().numpy()
        savez_compressed(paths[2][:-4], indices=int16(indices), correlation=corr)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    # path = 'data/nerf_llff_data/*/images_4/*.png'  # LLFF
    # path = 'data/Multiview-Segmentation-Data/images/*/images_?/*.png'  # MVSeg
    # path = 'data/Replica_Dataset/*_?/Sequence_1/rgb/*.png'  # Replica
    # path = 'data/360/*/images_?/*.JPG'  # 360

    # -- Validation -- #
    # path = '/media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/Segment/replica_sam/Replica640/room_0/rgb/*.png'  # Replica
    
    # path = '/media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/ParseNeRF/data/pigs/BamaPig3D_pure_pickle/select_images/1625/rgb/*.jpg'  # Replica
    path = '/media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/ParseNeRF/data/haoxiang/test_/rgb_4/*.png'
    
    files = sorted(glob(path))

    # N = environ.get('CUDA_VISIBLE_DEVICES', default=',,,,,,,').count(',') + 1
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # with ThreadPoolExecutor(N) as executor:
    #     executor.map(process_images, range(N), [files[n::N] for n in range(N)])
    
    # for f in files:
    process_images(0, files)
