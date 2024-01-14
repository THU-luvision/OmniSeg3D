import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.depth_flag = self.hparams.depth_flag
        self.semantic_flag = self.hparams.semantic_flag
        self.patch_flag = self.hparams.patch_flag

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        network_kwargs = {
            'semantic_flag': self.hparams.semantic_flag,
            'semantic_dim':  self.hparams.semantic_dim,
            'semantic_only': self.hparams.semantic_only,
            'device':        self.device,
            'aabb_tol':      self.hparams.aabb_tol,
        }
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, **network_kwargs)

        loss_kwargs = {
            'depth_flag':     self.hparams.depth_flag,
            'semantic_flag':  self.hparams.semantic_flag,
            'patch_flag':     self.hparams.patch_flag,
            'semantic_dim':   self.hparams.semantic_dim,
            'semantic_only':  self.hparams.semantic_only,
            'min_pixnum':  self.hparams.min_pixnum,
        }
        self.loss = NeRFLoss(lambda_opacity    = self.hparams.opacity_loss_w,
                             lambda_distortion = self.hparams.distortion_loss_w,
                             lambda_sam        = self.hparams.sam_loss_w,
                             sam_level_weight  = self.hparams.sam_level_w,
                             lambda_depth      = self.hparams.depth_loss_w,
                             **loss_kwargs)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time':      split!='train',
                  'bg_color':      self.hparams.bg_color,
                  'semantic_flag':  self.semantic_flag,
                  'semantic_dim':   self.hparams.semantic_dim,
                  'semantic_only':  self.hparams.semantic_only,
                  }
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir':       self.hparams.root_dir,
                  'downsample':     self.hparams.downsample,
                  'depth_flag':     self.depth_flag,
                  'semantic_flag':  self.semantic_flag,
                  'patch_flag':     self.patch_flag,
                  'semantic_path':  self.hparams.semantic_path,
                  }
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        if hasattr(self.train_dataset, 'xyz_min') and hasattr(self.train_dataset, 'xyz_max'):
            self.model.aabb_min = self.train_dataset.xyz_min.to(self.device) if \
                isinstance(self.train_dataset.xyz_min, torch.Tensor) else \
                torch.tensor(self.train_dataset.xyz_min, device=self.device)
            self.model.aabb_max = self.train_dataset.xyz_max.to(self.device) if \
                isinstance(self.train_dataset.xyz_max, torch.Tensor) else \
                torch.tensor(self.train_dataset.xyz_max, device=self.device)
        self.test_dataset = dataset(split='test', **kwargs)

        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

    def configure_optimizers(self):

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        excluded = ['dR', 'dT']
        if self.hparams.semantic_only:
            self.model.requires_grad_(False)
            self.model.zero_grad(True)
            self.model.sam_encode_net.requires_grad_(True)
            excluded += ['model.xyz_encoder.params', 'model.sigma_net.params',
                         'model.dir_encoder.params', 'model.rgb_net.params']
        for n, p in self.named_parameters():
            if n not in excluded: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=8,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        if not self.hparams.semantic_only:
            self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                            self.poses,
                                            self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0 and not self.hparams.semantic_only:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch, **{'global_step': self.global_step, 
                                              'ray_sampling_mixture': self.hparams.ray_sampling_strategy == "mixture"})

        if self.hparams.use_exposure and not self.hparams.semantic_only:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        if not self.hparams.semantic_only:
            with torch.no_grad():
                self.train_psnr(results['rgb'], batch['rgb'])
            self.log('train/psnr', self.train_psnr, True)
            # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        if self.semantic_flag:
            if 'semantic_render' in loss_d.keys():
                self.log('sam_loss', loss_d['semantic_render'], True)
            if 'sem_phydist' in loss_d.keys():
                self.log('sem_phydist', loss_d['sem_phydist'], True)



        # --- validate all test views: for incremental evaluation --- #
        if self.hparams.temporal_validate > 0:
            if self.global_step % self.hparams.temporal_validate == 0:
                metric_list = []
                for vi in range(20):
                    batch = self.test_dataset[vi]
                    for x in batch.keys():
                        if x in ["pose", "rgb", "depth"]:
                            batch[x] = batch[x].cuda()
                    
                    results = self(batch, split='test')
            
                    w, h = self.train_dataset.img_wh
                    if "rgb" in batch.keys():
                        rgb_gt = batch['rgb']
                        logs = []
                        # compute each metric per image
                        self.val_psnr(results['rgb'], rgb_gt)
                        logs.append(self.val_psnr.compute().cpu().numpy())
                        self.val_psnr.reset()

                        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
                        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
                        self.val_ssim(rgb_pred, rgb_gt)
                        logs.append(self.val_ssim.compute().cpu().numpy())
                        self.val_ssim.reset()
                        if self.hparams.eval_lpips:
                            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                                        torch.clip(rgb_gt*2-1, -1, 1))
                            logs.append(self.val_lpips.compute().cpu().numpy())
                            self.val_lpips.reset()
                        metric_list.append(logs)

                    if not self.hparams.no_save_test: # save test image to disk
                        self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
                        os.makedirs(self.val_dir, exist_ok=True)
                        idx = batch['img_idxs']
                        rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                        rgb_pred = (rgb_pred*255).astype(np.uint8)
                        
                        # depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                        imageio.imsave(os.path.join(self.val_dir, f'val_c_{self.global_step:04d}_{idx:03d}.png'), rgb_pred)
                        # imageio.imsave(os.path.join(self.val_dir, f'val_d_{self.global_step:04d}_{idx:03d}.png'), depth)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        self.test_dataset.split = 'test'
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/results'
            os.makedirs(self.val_dir, exist_ok=True)
            self.val_subdir_sam = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/results/sam'
            os.makedirs(self.val_subdir_sam, exist_ok=True)
            self.val_subdir_xyz = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/results/xyz'
            os.makedirs(self.val_subdir_xyz, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

            if self.semantic_flag:

                @torch.cuda.amp.autocast(False)
                def pca(X: torch.Tensor, n_components: int = 3) -> torch.Tensor:
                    X -= X.mean(0)
                    eigvals, eigvecs = torch.linalg.eigh(X.T @ X)
                    proj_mat = eigvecs[:, eigvals.topk(n_components)[1]]
                    return X @ proj_mat

                sam_norm = results['semantic'] / (results['semantic'].norm(dim=1, keepdim=True) + 1e-6)

                sam_norm = sam_norm * 0.5 + 0.5

                # --- PCA --- #
                sam_img = pca(sam_norm, n_components=3)
                sam_img = sam_img.cpu().numpy()

                sam_img = rearrange(sam_img, '(h w) c -> h w c', h=h)
                sam_img = (sam_img - sam_img.reshape(-1, 3).min(0))/ (sam_img.reshape(-1, 3).max(0) - sam_img.reshape(-1, 3).min(0))
                sam_img = np.uint8((sam_img - sam_img.min())/(sam_img.max() - sam_img.min())*255)
                imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_s.png'), sam_img)

                raw_sam_img = rearrange(results['semantic'].cpu().numpy(), '(h w) c -> h w c', h=h)
                np.save(os.path.join(self.val_subdir_sam, f'{idx:03d}.npy'), raw_sam_img.astype(np.float16))
                xyz_img = rearrange(results['xyz'].cpu().numpy(), '(h w) c -> h w c', h=h)
                np.save(os.path.join(self.val_subdir_xyz, f'{idx:03d}.npy'), xyz_img)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('high')    # add by yixuan (torch > 2.0)

    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    if hparams.semantic_only and (not hparams.weight_path):
        raise ValueError('You need to provide a @weight_path for semantic-only training!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'results/{hparams.dataset_name}/{hparams.exp_name}/ckpts',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"results/{hparams.dataset_name}",
                               name=hparams.exp_name + "/logs",
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPStrategy(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    # trainer.fit(system, ckpt_path=hparams.ckpt_path)
    if not hparams.val_only:
        trainer.fit(system, ckpt_path=hparams.ckpt_path)
    else:
        trainer.validate(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'results/{hparams.dataset_name}/{hparams.exp_name}/ckpts/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'results/{hparams.dataset_name}/{hparams.exp_name}/ckpts/epoch={hparams.num_epochs-1}_slim.ckpt')

