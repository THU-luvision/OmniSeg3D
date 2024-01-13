from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        # self.crop = True

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
                img_idxs_one = img_idxs
            elif self.ray_sampling_strategy == 'mixture':
                # --- all + same --- #
                img_idxs_all = np.random.choice(len(self.poses), self.batch_size // 2)
                img_idxs_one = np.random.choice(len(self.poses), 1)[0]
                img_idxs_same = np.repeat(img_idxs_one, repeats=self.batch_size // 2)
                img_idxs = np.concatenate([img_idxs_all, img_idxs_same])

            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size, False)
            # if self.crop is True:
            #     # x:50:1870 y: 140:1000
            #     cb = np.array([50, 1870, 140, 1000])    # crop boundary
            #     xlen = cb[1] - cb[0]
            #     ylen = cb[3] - cb[2]
            #     x = np.arange(cb[0], cb[1])[np.random.choice(xlen, self.batch_size, True)]
            #     y = np.arange(cb[2], cb[3])[np.random.choice(ylen, self.batch_size, True)]
            #     pix_idxs = x + y*1920
            #     # pix_idxs = y + x*ylen
            # else:
            #     pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size, False)
            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}
            # if self.rays.shape[-1] == 4: # HDR-NeRF data
            #     sample['exposure'] = rays[:, 3:]
            if (self.rays.shape[-1] >= 4):
                if 'depth' in self.mode.keys(): # depth data
                    sample['depth'] = rays[:, self.mode['depth']]
                if 'sam' in self.mode.keys():
                    sample['sam'] = rays[:, self.mode['sam']]
                    if self.patch_flag:
                        sample['accpatch'] = self.patchmaps[img_idxs_one]

        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample