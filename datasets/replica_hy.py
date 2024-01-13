# --- borrowed from /media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/ngp_pl_0216/PARF_backup_0601/ngp_pl_backup_0529_for_reproduce_Replica

import torch
import json
import glob
import numpy as np
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from .ray_utils import get_ray_directions, get_rays
from .color_utils import read_image, read_depth

from .base import BaseDataset

import imageio, cv2
# from matplotlib import pyplot as plt
# from einops import rearrange

import scipy


class ReplicaDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)


        self.transform_fname = "transforms_plane.json"


        self.depth_flag = kwargs.get('depth_flag', False)
        self.semantic_flag = kwargs.get('sam_flag', False)
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

            self.imgs_lis = sorted(glob.glob(os.path.join(self.root_dir, 'release/rgb/*.jpg')))
            self.deps_lis = sorted(glob.glob(os.path.join(self.root_dir, 'release/depth/*.png')))
            if self.patch_flag:
                self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'release/sam/*.npz')))
            else:
                self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'release/sam_raw/*.png')))

            # --- load RGB, Depth, Semantics --- #
            rays, self.patchmaps = zip(*thread_map(self.read_meta, idx_list))

            self.rays = torch.stack(rays)
            self.rays[..., self.mode['depth']] /= self.scale     # depth scale trans



    def read_meta(self, idx: int) -> tuple:
        image = imageio.imread(self.imgs_lis[idx])
        image = torch.tensor(image.astype(np.float32).reshape(-1, 3) / 255)
        depth = imageio.imread(self.deps_lis[idx])
        depth = torch.tensor(depth.astype(np.float32).reshape(-1, 1) * self.depthscale)
        if not self.semantic_flag:
            return torch.cat((image, depth), 1), None
        if not self.patch_flag:
            index = imageio.imread(self.sams_lis[idx])
            index = torch.tensor(index.reshape(-1, 1))
            return torch.cat((image, depth, index), 1), None
        sam = np.load(self.sams_lis[idx])
        index = torch.tensor(sam['indices'].reshape(-1, 1))
        return torch.cat((image, depth, index), 1), sam['correlation']



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


    


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))


def getGaussKernel(sigma, H, W):
    r, c = np.mgrid[0:H:1, 0:W:1]
    r = r - (H - 1) / 2
    c = c - (W - 1) / 2
    gaussMatrix = np.exp(-0.5 * (np.power(r, 2) + np.power(c, 2)) / np.power(sigma, 2))
    # 计算高斯矩阵的和
    sunGM = np.sum(gaussMatrix)
    # 归一化
    # gaussKernel = gaussMatrix / sunGM
    gaussKernel = (gaussMatrix - gaussMatrix.min()) / (gaussMatrix.max() - gaussMatrix.min())
    return gaussKernel


def render_pose_axis(poses, unit_length=0.1):
    """
    poses: c2w matrix in opencv manner
    unit_length (m:meter): unit axis length for visualization
    axis-x, axis-y, axis-z: red, green, blue
    """
    pose_coord_x = poses[:, :3, -1] + poses[:, :3, 0] * unit_length
    pose_coord_y = poses[:, :3, -1] + poses[:, :3, 1] * unit_length
    pose_coord_z = poses[:, :3, -1] + poses[:, :3, 2] * unit_length
    poses_vis = np.concatenate([poses[:, :3, -1], pose_coord_x, pose_coord_y, pose_coord_z], axis=0)
    poses_rgb = np.concatenate([np.ones([poses.shape[0], 3])*255,
                                np.ones([poses.shape[0], 3])*np.array([255, 0, 0]),
                                np.ones([poses.shape[0], 3])*np.array([0, 255, 0]),
                                np.ones([poses.shape[0], 3])*np.array([0, 0, 255]),
                                ])
    # pcwrite("camera_raw_axis.ply", np.concatenate([poses_vis, poses_rgb], axis=1))
    return poses_vis, poses_rgb