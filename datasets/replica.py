# --- borrowed from /media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/ngp_pl_0216/PARF_backup_0601/ngp_pl_backup_0529_for_reproduce_Replica

import torch
import json
import glob
import numpy as np
import os
import imageio
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from .ray_utils import get_ray_directions, get_rays
from .color_utils import read_image, read_depth

from .base import BaseDataset



class ReplicaDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)


        self.transform_fname = "transforms_train.json"


        self.depth_flag = kwargs.get('depth_flag', False)
        self.semantic_flag = kwargs.get('semantic_flag', False)
        self.patch_flag = kwargs.get('patch_flag', False)

        # --- mode setting --- #
        self.mode = {'rgb':[0,1,2]}
        if self.depth_flag and self.semantic_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3], 'sam':[4]}
        elif self.depth_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3]}
        elif self.semantic_flag:
            self.mode = {'rgb':[0,1,2], 'sam':[3]}
            
        self.read_intrinsics(downsample=downsample)

        if kwargs.get('read_meta', True):

            if split == "train":
                step = 20
                fids = np.arange(0, 2000, step)
                idx_list = np.arange(0, 2000//10, step//10)
            if split == "test":
                step = 80
                fids = np.arange(10, 2000, step)
                idx_list = np.arange(1, 2000//10, step//10)

            # --- load poses --- #
            self.poses = []
            with open(os.path.join(self.root_dir, self.transform_fname), 'r') as f:
                frames = json.load(f)["frames"]
                for frame in frames:
                    c2w = np.array(frame['transform_matrix'])[:3, :4]   # (right down front)
                    c2w[:, 3] -= self.shift
                    c2w[:, 3] /= self.scale # to bound the scene inside [-0.5, 0.5]
                    self.poses += [c2w]
            self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
            self.poses = self.poses[fids, ...]

            self.imgs_lis = sorted(glob.glob(os.path.join(self.root_dir, 'results/rgb/*.jpg')))
            self.deps_lis = sorted(glob.glob(os.path.join(self.root_dir, 'results/depth/*.png')))
            if self.patch_flag:
                self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'results/sam/*.npz')))
            else:
                self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'results/sam_raw/*.png')))

            # --- load RGB, Depth, Semantics --- #
            rays, self.patchmaps = zip(*thread_map(self.read_meta, idx_list))

            self.rays = torch.stack(rays)
            if self.depth_flag:
                self.rays[..., self.mode['depth']] /= self.scale     # depth scale trans



    def read_meta(self, idx: int) -> tuple:
        image = imageio.imread(self.imgs_lis[idx])
        image = torch.tensor(image.astype(np.float32).reshape(-1, 3) / 255)
        if self.depth_flag:
            depth = imageio.imread(self.deps_lis[idx])
            depth = torch.tensor(depth.astype(np.float32).reshape(-1, 1) * self.depthscale)
            image = torch.cat((image, depth), 1)
        if not self.semantic_flag:
            return image, None
        else:
            if not self.patch_flag:
                index = imageio.imread(self.sams_lis[idx])
                index = torch.tensor(index.reshape(-1, 1))
                image = torch.cat((image, index), 1)
                return image, None
            else:
                sam = np.load(self.sams_lis[idx])
                index = torch.tensor(sam['indices'].reshape(-1, 1))
                image = torch.cat((image, index), 1)
                return image, sam['correlation']



    def read_intrinsics(self, downsample=1.0):
        with open(os.path.join(self.root_dir, self.transform_fname), 'r') as f:
            meta = json.load(f)
        
        xyz_min, xyz_max = np.array(meta["aabb"])   # bbox shape: (2, 3)
        self.shift = (xyz_max+xyz_min)/2
        self.scale = (xyz_max-xyz_min).max() * 1.2 # enlarge a little
        
        w = int(meta["w"] * downsample) 
        h = int(meta["h"] * downsample)
        fx = meta["fl_x"] * downsample
        fy = meta["fl_y"] * downsample
        
        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

        self.depthscale = meta["integer_depth_scale"]   # 1/6553.5
