import torch
from torch import nn
import vren

sim = lambda x, y, l : torch.exp(-l * (x - y)**2)

class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, 
            lambda_opacity=1e-3, 
            lambda_distortion=1e-3, 
            lambda_sam=1e-4, 
            sam_level_weight=5, 
            lambda_depth=1.,
            **kwargs
        ):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.lambda_sam = lambda_sam
        self.sam_level_weight = sam_level_weight
        self.lambda_depth = lambda_depth

        self.depth_flag = kwargs.get('depth_flag', False)
        self.semantic_flag = kwargs.get('semantic_flag', False)
        self.semantic_dim = kwargs.get('semantic_dim', False)
        self.semantic_only = kwargs.get('semantic_only', False)
        self.patch_flag = kwargs.get('patch_flag', False)
        self.min_pixnum = kwargs.get('min_pixnum', 2)

        self.normalize_sam = False
        self.sam_norm_loss_flag = True
        self.objsize_schedule = False

    def forward(self, results, target, **kwargs):
        if 'global_step' in kwargs:
            global_step = kwargs['global_step']
        ray_sampling_mixture = kwargs.get('ray_sampling_mixture', False)
        
        batchsize = target['rgb'].shape[0]

        d = {}
        if not self.semantic_only:
            if ray_sampling_mixture:
                d['rgb'] = ((results['rgb']-target['rgb'])**2)[:batchsize//2, :]
            else:
                d['rgb'] = (results['rgb']-target['rgb'])**2
            
            o = results['opacity']+1e-10
            # encourage opacity to be either 0 or 1 to avoid floater
            if self.lambda_opacity > 0:
                d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

            if self.lambda_distortion > 0:
                d['distortion'] = self.lambda_distortion * \
                    DistortionLoss.apply(results['ws'], results['deltas'],
                                        results['ts'], results['rays_a'])
            
            if self.depth_flag and "depth" in target.keys():
                valid_depth_idx = target['depth'] > 0
                d['depth_render'] = self.lambda_depth * (results['depth'][valid_depth_idx.squeeze()]-target['depth'][valid_depth_idx])**2


        if self.semantic_flag and "sam" in target.keys():

            if self.objsize_schedule:
                self.min_pixnum = int((1 - global_step / 5000) * (20-3) + 3)    # iter:0~5000, minpix: 20~3

            if ray_sampling_mixture:
                # --- mixture sampling --- #
                valid_semantic_idx = target['sam'][batchsize//2:, :] > 0
                sam_t = target['sam'][batchsize//2:, :][valid_semantic_idx].long()
                sam_o = results['semantic'][batchsize//2:, :][valid_semantic_idx.squeeze(), :]
            else:
                valid_semantic_idx = target['sam'] > 0
                sam_t = target['sam'][valid_semantic_idx].long()
                sam_o = results['semantic'][valid_semantic_idx.squeeze(), :]
            
            if self.normalize_sam:
                sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6)
            

            ## --- Contructive Clustering --- #
            cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
            cluster_ids = cluster_ids[cnums_all > self.min_pixnum]
            cnums = cnums_all[cnums_all > self.min_pixnum]
            cnum = cluster_ids.shape[0] # cluster number

            u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
            phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


            for i in range(cnum):
                cluster = sam_o[sam_t == cluster_ids[i], :]
                u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
                phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))

            if self.patch_flag:
                accpatch = target['accpatch']


            # tau = 0.1; phi_list[:, 0] = tau    # option 1: constant temperature
            # phi_list = phi_list * (tau / phi_list.mean())     # option 2: (PCL) too small phi causes too large num in torch.exp().
            # phi_list = (phi_list - phi_list.min()) / (phi_list.max() - phi_list.min()) * 5 + 0.1   # scale to range [0.1, 5.1]
            phi_list = torch.clip(phi_list * 10, min=0.1, max=1.0)
            phi_list = phi_list.detach()
            
            ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

            for i in range(cnum):
                cluster = sam_o[sam_t == cluster_ids[i], :]

                dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]

                if not self.patch_flag:

                    ProtoNCE += -torch.sum(torch.log(
                        dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                        ))

                else:
                    acc_list = accpatch[cluster_ids[i], :]
                    acc_list_c = acc_list[cluster_ids]  # for the clusters only
                    acc_h_cnts = torch.sort(torch.unique(acc_list_c), descending=True).values[:-1]    # remove last one "0"
                    levelnum = acc_h_cnts.shape[0]
                    
                    for l, level in enumerate(acc_h_cnts):
                        level_cids = torch.argwhere(acc_list_c == level).squeeze()   # cluster ids on each level

                        cal_opt = 3
                        
                        if cal_opt == 1:
                            # --- option 1: mean patches dist
                            dist_mean = dist[:, level_cids].reshape(dist.shape[0], -1).mean(dim=1, keepdim=True)
                            tmp_loss = -torch.sum(torch.log(
                                    dist_mean / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                                ))
                            max_loss = tmp_loss if l == 0 else max(tmp_loss, max_loss)  # --- unidirectional hierarchical loss --- #

                        elif cal_opt == 2:
                            # --- option 2: all patches dist
                            dist_patches = dist[:, level_cids].reshape(dist.shape[0], -1)
                            tmp_loss = -torch.sum(
                                    torch.log(
                                        dist_patches / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                                    ).mean(dim=1)
                                )
                            max_loss = tmp_loss if l == 0 else max(tmp_loss, max_loss)  # --- unidirectional hierarchical loss --- #
                        
                        elif cal_opt == 3:
                            # --- option 3: per pixel patches dist,  unidirectional hierarchical loss --- #
                            dist_patches = dist[:, level_cids].reshape(dist.shape[0], -1)

                            log = - torch.log(
                                    dist_patches / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)     # (pixnum, L_patchnum)
                                )
                            if l == 0:
                                max_loss = torch.sum( log.mean(dim=1) )
                                # --- set max_log for each pix --- #
                                max_log = torch.max(log, dim=1, keepdim=True).values   # patch with max lose (pixnum, 1)
                            else:
                                # --- thres by the last layer --- #
                                log_thres = torch.where(log > max_log, log, max_log)
                                max_loss = torch.sum( log_thres.mean(dim=1) )
                                # --- update max_log for each pix --- #
                                max_log = torch.max(log_thres, dim=1, keepdim=True).values   # patch with max lose (pixnum, 1)
                        
                        # ProtoNCE += max_loss
                        # ProtoNCE += max_loss / levelnum
                        ProtoNCE += max_loss * self.sam_level_weight**(-l)     # --- level weighting --- #

            d['semantic_render'] = self.lambda_sam * ProtoNCE

            if self.sam_norm_loss_flag:
                sam_norm_loss = ((torch.norm(sam_o, dim=-1, keepdim=True) - 1.0) ** 2).mean()
                d['semantic_render'] += 100 * sam_norm_loss


        return d
