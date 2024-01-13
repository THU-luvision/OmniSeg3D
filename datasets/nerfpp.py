import torch
import glob
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class NeRFPPDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        K = np.loadtxt(glob.glob(os.path.join(self.root_dir, 'train/intrinsics/*.txt'))[0],
                       dtype=np.float32).reshape(4, 4)[:3, :3]
        K[:2] *= self.downsample
        w, h = Image.open(glob.glob(os.path.join(self.root_dir, 'train/rgb/*'))[0]).size
        w, h = int(w*self.downsample), int(h*self.downsample)
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'test_traj':
            poses_path = \
                sorted(glob.glob(os.path.join(self.root_dir, 'camera_path/pose/*.txt')))
            self.poses = [np.loadtxt(p).reshape(4, 4)[:3] for p in poses_path]
        else:
            if split=='trainval':
                img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'train/rgb/*')))+\
                            sorted(glob.glob(os.path.join(self.root_dir, 'val/rgb/*')))
                poses = sorted(glob.glob(os.path.join(self.root_dir, 'train/pose/*.txt')))+\
                       sorted(glob.glob(os.path.join(self.root_dir, 'val/pose/*.txt')))
            else:
                img_paths = sorted(glob.glob(os.path.join(self.root_dir, split, 'rgb/*')))
                poses = sorted(glob.glob(os.path.join(self.root_dir, split, 'pose/*.txt')))

            print(f'Loading {len(img_paths)} {split} images ...')
            for img_path, pose in tqdm(zip(img_paths, poses)):
                self.poses += [np.loadtxt(pose).reshape(4, 4)[:3]]

                img = read_image(img_path, self.img_wh)
                self.rays += [img]

            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
