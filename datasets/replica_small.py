from imageio import imread
import numpy as np
import torch
from tqdm.contrib.concurrent import thread_map

from .base import BaseDataset
from .ray_utils import get_ray_directions


class ReplicaSmallDataset(BaseDataset):
    def __init__(self, root_dir: str, **kwargs) -> None:
        super().__init__(root_dir)
        self.depth_flag = kwargs.get('depth_flag', True)
        self.semantic_flag = kwargs.get('semantic_flag', False)
        self.patch_flag = kwargs.get('patch_flag', True)
        assert self.depth_flag
        self.mode = {'rgb': [0, 1, 2], 'depth': [3]}
        if self.semantic_flag: self.mode['sam'] = [4]
        self.path = kwargs.get('semantic_path', 'sam')

        self.img_wh = (640, 480)
        self.K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
        self.directions = get_ray_directions(480, 640, self.K)
        poses = np.loadtxt(f'{root_dir}/traj_w_c.txt', np.float32)
        self.poses = torch.tensor(poses.reshape(-1, 4, 4)[:, :3])

        frames = np.arange(900)
        if 'office_1' in root_dir: frames = np.delete(frames, np.s_[474:504])
        elif 'office_4' in root_dir: frames = np.delete(frames, np.s_[618:734])
        frames = np.arange(0, 900, 10)
        rays, self.patchmaps, bounds = zip(*thread_map(self.read_meta, frames))
        self.rays, bounds = torch.stack(rays), torch.stack(bounds)
        bounds = torch.stack((bounds[:, 0].amin(0), bounds[:, 1].amax(0)))
        self.rays[..., 3] /= (scale := 1.1 * (bounds[1] - bounds[0]).max())     # depth scale trans

        self.poses = self.poses[torch.tensor(frames)]
        self.poses[..., 3] -= (center := bounds.mean(0))
        self.poses[..., 3] /= scale
        self.xyz_min, self.xyz_max = (bounds - center) / scale

    # def read_meta(self, frame: int) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    def read_meta(self, frame: int) -> tuple:
        image = imread(f'{self.root_dir}/rgb/rgb_{frame}.png')
        image = torch.tensor(image.astype(np.float32).reshape(-1, 3) / 255)
        depth = imread(f'{self.root_dir}/depth/depth_{frame}.png')
        depth = torch.tensor(depth.astype(np.float32).reshape(-1, 1) / 1000)
        xyz = self.poses[frame, :, 3] + self.directions * depth
        bound = torch.stack((xyz.amin(0), xyz.amax(0)))
        if not self.semantic_flag:
            return torch.cat((image, depth), 1), None, bound
        if not self.patch_flag:
            index = imread(f'{self.root_dir}/sam_raw/sam_raw_{frame}.png')
            index = torch.tensor(index.reshape(-1, 1))
            return torch.cat((image, depth, index), 1), None, bound
        sam = np.load(f'{self.root_dir}/{self.path}/{self.path}_{frame}.npz')
        index = torch.tensor(sam['indices'].reshape(-1, 1))
        return torch.cat((image, depth, index), 1), sam['correlation'], bound
