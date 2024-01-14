import torch
import numpy as np
import os
from os.path import dirname, basename, join
import glob
from tqdm import tqdm
from cv2 import resize, INTER_AREA, INTER_NEAREST
import trimesh

from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary, read_array

from .base import BaseDataset


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        self.depth_flag = kwargs.get('depth_flag', False)
        self.semantic_flag = kwargs.get('semantic_flag', False)
        self.patch_flag = kwargs.get('patch_flag', False)
        assert (self.patch_flag or not self.semantic_flag), 'only patch mode is supported'

        if self.depth_flag and self.semantic_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3], 'sam':[4]}
        elif self.depth_flag:
            self.mode = {'rgb':[0,1,2], 'depth':[3]}
        elif self.semantic_flag:
            self.mode = {'rgb':[0,1,2], 'sam':[3]}
            self.sam_folder = glob.glob(join(self.root_dir, "sam*"))[0]
            print("DATA_LOADER: load sam files from: ", self.sam_folder)

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = round(camdata[1].height*self.downsample)
        w = round(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        folder = f'images_{round(1 / self.downsample)}'
        if not os.path.isdir(os.path.join(self.root_dir, folder)):
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, folder, '*.*p*g')) +
                           glob.glob(os.path.join(self.root_dir, folder, '*.*P*G')))
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats[perm]) # (N_images, 4, 4) cam2world matrices

        if os.path.isfile(os.path.join(self.root_dir, 'sparse/0/roi.ply')):
            pcd = trimesh.load(os.path.join(self.root_dir, 'sparse/0/roi.ply'))
            to_origin, extents = trimesh.bounds.oriented_bounds(pcd)
            self.poses = (to_origin @ poses)[:, :3]
            scale = extents.max()
            self.poses[..., 3] /= scale
            self.xyz_min, self.xyz_max = -.5 * extents / scale, .5 * extents / scale
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
            pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

            self.poses, self.pts3d = center_poses(poses[:, :3], pts3d)

            scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
            self.poses[..., 3] /= scale
            self.pts3d /= scale

        self.rays = []
        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        if self.semantic_flag:
            self.patchmaps = []

        if 'llff' in self.root_dir: # LLFF data
            test_idx = 1 if 'fern' in self.root_dir \
                else 8 if 'horns' in self.root_dir \
                else 12 if 'orchids' in self.root_dir \
                else 31 if 'trex' in self.root_dir \
                else 0 # consistent with NVOS
            if split=='train':
                img_paths.pop(test_idx)
                self.poses = np.delete(self.poses, test_idx, 0)
        # elif 'Multiview-Segmentation-Data' in self.root_dir: # SPIn-NeRF data
        #     # use all images for training
        #     pass
        # elif '360' in self.root_dir: # 360 data
        #     # use all images for training
        #     pass
        else:
            # use every 8th image as test set
            if split=='train':
                img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
            elif split=='test':
                img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            buf = [] # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]

            if self.depth_flag:
                depth = read_array(img_path.replace(folder, 'stereo/depth_maps') + '.geometric.bin')
                depth = resize(depth, self.img_wh, interpolation=INTER_AREA) / scale
                buf.append(torch.FloatTensor(depth.reshape(-1, 1)))

            if self.semantic_flag:
                # sam = np.load(img_path.replace('images_', 'sam_')[:-3] + 'npz')
                # sam = np.load(join(dirname(dirname(img_path)), 'sam', basename(img_path).split('.')[0]+'.npz'))
                sam = np.load(join(self.sam_folder, basename(img_path).split('.')[0]+'.npz'))
                indices = resize(sam['indices'], self.img_wh, interpolation=INTER_NEAREST)
                buf.append(torch.FloatTensor(indices.reshape(-1, 1)))
                self.patchmaps.append(sam['correlation'])

            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)