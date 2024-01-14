import torch
import json
import glob
import numpy as np
import os
import tqdm

from .ray_utils import get_ray_directions, get_rays
from .color_utils import read_image, read_depth

from .base import BaseDataset

import imageio, cv2
import scipy


class MVSNetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics(downsample=downsample)

        self.depth_flag = kwargs.get('depth_flag', False)
        self.semantic_flag = kwargs.get('sam_flag', False)
        self.patch_flag = kwargs.get('patch_flag', False)

        # --- mode setting --- #
        self.mode = {'rgb':[0,1,2]}; cnt = 3
        # self.mode = {'rgb':[0,1,2, 3]}; cnt = 4     # with_mask
        if self.depth_flag:
            self.mode['depth'] = [cnt]; cnt += 1
        if self.semantic_flag:
            self.mode['sam'] = [cnt]; cnt += 1

        if kwargs.get('read_meta', True):
            self.read_meta(split)


    def read_intrinsics(self, downsample=1.0):
        
        xyz_min, xyz_max = np.loadtxt(os.path.join(self.root_dir, "aabb.txt"))   # (2, 3) bbox
        self.shift = (xyz_max+xyz_min)/2                                # array([-0.34171931, -0.16075864, -0.71072144])
        self.scale = (xyz_max-xyz_min).max()/2 * 1.2 # enlarge a little   # 5.13558

        intrinsics = np.loadtxt(os.path.join(self.root_dir, "intrinsics.txt"))  # (3, 3) mvsnet format
    
        K = intrinsics * downsample
        w = int(K[0, 2] * 2)
        h = int(K[1, 2] * 2)
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

        self.depthscale = 1 / 1000


    def read_meta(self, split):
        self.rays = []
        self.poses = []

        imgs_f = sorted(glob.glob(os.path.join(self.root_dir, "images_4/*")))
        cams_f = sorted(glob.glob(os.path.join(self.root_dir, "cams_4/*")))
        msks_f = sorted(glob.glob(os.path.join(self.root_dir, "mask/*")))
        imgs_num = len(imgs_f)

        if split == "train":
            step = 1
            idx_list = np.arange(0, imgs_num, step)
        if split == "test":
            step = 20
            idx_list = np.arange(3, imgs_num, step)

        self.n_images = idx_list.shape[0]

        self.imgs_lis = [imgs_f[i] for i in idx_list]
        self.cams_lis = [cams_f[i] for i in idx_list]
        self.msks_lis = [msks_f[i] for i in idx_list]
        if self.depth_flag:
            self.deps_lis = sorted(glob.glob(os.path.join(self.root_dir, 'depth_4/*.png')))
            self.deps_lis = [self.deps_lis[i] for i in idx_list]
        if self.semantic_flag:
            if not self.patch_flag:
                self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'sam_5/sam_5_standard/*.png')))
            else:
                self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'sam_5/sam_patch/*.png')))
                self.patchmaps_lis = sorted(glob.glob(os.path.join(self.root_dir, 'sam_5/sam_patch_map/*.npz')))
                self.patchmaps = []
                self.patchmaps_lis = [self.patchmaps_lis[i] for i in idx_list]
            self.sams_lis = [self.sams_lis[i] for i in idx_list]

        print(f'Loading {self.n_images} {split} images ...')

        for i in tqdm.trange(self.n_images):

            with open(self.cams_lis[i]) as f:
                lines = f.readlines()
            cameraRTO = np.eye(4).astype(np.float32)
            cameraRTO[0, :] = np.array(lines[1].rstrip().split(' ')[:4], dtype=np.float32)
            cameraRTO[1, :] = np.array(lines[2].rstrip().split(' ')[:4], dtype=np.float32)
            cameraRTO[2, :] = np.array(lines[3].rstrip().split(' ')[:4], dtype=np.float32)

            c2w = np.linalg.inv(cameraRTO)[:3, :4]   # (right down front)
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
            self.poses += [c2w]

            img = read_image(self.imgs_lis[i], img_wh=self.img_wh)
            # msk = read_image(self.msks_lis[i], img_wh=self.img_wh)[:, 0] / 255.0
            
            if self.depth_flag:
                dep = read_depth(self.deps_lis[i], self.img_wh, self.depthscale)

                debug = False
                if debug:
                    dep2 = read_depth(self.deps_lis[i], self.img_wh, self.depthscale)
                    pts = self.directions * dep2
                    pose = np.linalg.inv(cameraRTO)[:3, :4]
                    pts2 = np.concatenate([pts, np.ones(pts.shape[0])[:, None]], axis=1)    # (h*w, 3) -> (h*w, 4)
                    pts2w = pts2 @ pose.transpose(1, 0)
                    pcwrite("debug_depth_backproj.ply", np.concatenate([pts2w, img*255], axis=1))

                dep = dep / (2*self.scale)    # world scale
                img = np.concatenate([img, dep], axis=-1)

            
            if self.semantic_flag:
                sam = cv2.imread(self.sams_lis[i], -1).reshape(self.img_wh[0]*self.img_wh[1], 1)
                img = np.concatenate([img, sam], axis=-1)

                if self.patch_flag:
                    patchmap = np.load(self.patchmaps_lis[i])['accmaps']
                    self.patchmaps.append(patchmap)

            self.rays += [img]
        
        self.rays = np.stack(self.rays)
        self.poses = np.stack(self.poses)

        self.rays = torch.FloatTensor(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)


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