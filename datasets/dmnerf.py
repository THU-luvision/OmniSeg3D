import torch
import glob
import numpy as np
import os
from tqdm import tqdm
import cv2

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class DMNeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.depth_flag = kwargs.get('depth_flag', False)
        self.sam_flag = kwargs.get('sam_flag', False)
        
        # --- mode setting --- #
        self.mode = {'rgb':[0,1,2]}
        if self.depth_flag and self.sam_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3], 'sam':[4]}
        elif self.depth_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3]}
        elif self.sam_flag:
            self.mode = {'rgb':[0,1,2], 'sam':[3]}

        self.render_cameras_name = "cameras_sphere.npz"
        self.object_cameras_name = "cameras_sphere.npz"

        # self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.shift = 0.0
            self.scale = 2 # enlarge a ratio

            self.read_meta(split)


    # def read_intrinsics(self):
    #     K = np.loadtxt(glob.glob(os.path.join(self.root_dir, 'train/intrinsics/*.txt'))[0],
    #                    dtype=np.float32).reshape(4, 4)[:3, :3]
    #     K[:2] *= self.downsample
    #     w, h = Image.open(glob.glob(os.path.join(self.root_dir, 'train/rgb/*'))[0]).size
    #     w, h = int(w*self.downsample), int(h*self.downsample)
    #     self.K = torch.FloatTensor(K)
    #     self.directions = get_ray_directions(h, w, self.K)
    #     self.img_wh = (w, h)


    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train':
            idx_list = np.arange(0, 300, 1)
        elif split == 'test':
            idx_list = np.arange(1, 300, 8)


        # ------ load cameras ------ #
        camera_dict = np.load(os.path.join(self.root_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in idx_list]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in idx_list]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            # scale_mat[:3, :3] *= self.scale
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            pose[:3, 3] /= self.scale
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]

        # ------ load images ------ #
        self.images_lis = sorted(glob.glob(os.path.join(self.root_dir, 'image/*.png')))
        self.images_lis = [self.images_lis[i] for i in idx_list]
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) for im_name in self.images_lis]) / 256.0
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.img_wh = (self.W, self.H)
        self.rays += [self.images.reshape(self.n_images, self.H*self.W, -1)]


        # ------ load other images ------ #
        if self.depth_flag:
            self.depths_lis = sorted(glob.glob(os.path.join(self.root_dir, 'depth/*.png')))
            self.depths_lis = [self.depths_lis[i] for i in idx_list]
            self.depths_np = np.stack([cv2.imread(im_name, -1) for im_name in self.depths_lis]) / 1000.0
            # --- transform depth scale to unit sphere --- #
            inv_scale_mat = np.linalg.inv(self.scale_mats_np[0] * self.scale)
            self.depths_np = inv_scale_mat[0,0] * self.depths_np.astype(np.float32)  # [n_images, H, W]
            self.rays += [self.depths_np.reshape(self.n_images, self.H*self.W, 1)]
        
        if self.sam_flag:
            self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'sam/*.png')))
            # self.sams_lis = sorted(glob.glob(os.path.join(self.root_dir, 'semantic_instance/*.png')))
            self.sams_lis = [self.sams_lis[i] for i in idx_list]
            self.sams_np = np.stack([cv2.imread(im_name, -1) for im_name in self.sams_lis])
            self.rays += [self.sams_np.reshape(self.n_images, self.H*self.W, 1)]


        self.rays = torch.FloatTensor(np.concatenate(self.rays, axis=-1))
        self.poses = torch.FloatTensor(self.pose_all)


        # ------ load intrinsics ------ #
        self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]

        K = self.intrinsics_all[0, :3, :3]
        K[:2] *= self.downsample
        self.W, self.H = int(self.W*self.downsample), int(self.H*self.downsample)
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(self.H, self.W, self.K)



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