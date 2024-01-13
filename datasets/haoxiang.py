
import os
import json
from imageio import imread
import numpy as np
import torch
from tqdm.contrib.concurrent import thread_map

from .base import BaseDataset
from .ray_utils import get_ray_directions


class HaoxiangDataset(BaseDataset):
    def __init__(self, root_dir: str, **kwargs) -> None:
        super().__init__(root_dir)
        self.depth_flag = kwargs.get('depth_flag', False)
        self.semantic_flag = kwargs.get('semantic_flag', False)
        self.patch_flag = kwargs.get('patch_flag', True)
        downsample = kwargs.get('downsample', 1.0)

        # --- mode setting --- #
        self.mode = {'rgb':[0,1,2]}
        if self.depth_flag and self.semantic_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3], 'sam':[4]}
        elif self.depth_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3]}
        elif self.semantic_flag:
            self.mode = {'rgb':[0,1,2], 'sam':[3]}
        self.path = 'sam_4'


        self.transform_fname = "transforms_train.json"
        self.read_intrinsics(downsample=downsample)

        # self.img_wh = (4000, 3000)
        # self.K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
        # self.directions = get_ray_directions(480, 640, self.K)
        # poses = np.loadtxt(f'{root_dir}/traj_w_c.txt', np.float32)
        # self.poses = torch.tensor(poses.reshape(-1, 4, 4)[:, :3])

        self.poses = []
        with open(os.path.join(self.root_dir, self.transform_fname), 'r') as f:
            frames = json.load(f)["frames"]
            for frame in frames:
                c2w = torch.tensor(frame['transform_matrix'])[:3, :4]   # (right down front)
                self.poses += [c2w]
            self.poses = torch.stack(self.poses)

        frames = np.arange(self.poses.shape[0])

        rays, self.patchmaps = zip(*thread_map(self.read_meta, frames))
        self.rays = torch.stack(rays)

        self.poses = self.poses[torch.tensor(frames)]
        self.poses[..., 3] -= self.shift
        self.poses[..., 3] /= self.scale


    def read_intrinsics(self, downsample=1.0):
        with open(os.path.join(self.root_dir, self.transform_fname), 'r') as f:
            meta = json.load(f)
        
        xyz_min, xyz_max = np.array(meta["aabb"])   # (2, 3) bbox
        self.shift = (xyz_max+xyz_min)/2
        self.scale = (xyz_max-xyz_min).max()/2 * 4 # enlarge a little 
        
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


    def read_meta(self, frame: int) -> tuple:
        image = imread('{}/rgb_4/img_4-1{:0>4d}.png'.format(self.root_dir, frame))
        image = torch.tensor(image.astype(np.float32).reshape(-1, 3) / 255)
        sam = np.load('{}/{}/img_4-1{:0>4d}.npz'.format(self.root_dir, self.path, frame))
        index = torch.tensor(sam['indices'].reshape(-1, 1))
        return torch.cat((image, index), 1), sam['correlation']
