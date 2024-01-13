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
        ):  # depth_flag=False, semantic_flag=False
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

        self.normalize_sam = False
        self.sam_norm_loss_flag = True
        self.min_pixnum = 2
        self.objsize_schedule = False

        self.phydist_modes = ["None", "ratio_regularization", "order_loss", "phydist_contrast", "order_loss_supp", "patch-segment"]
        self.phydist_mode = self.phydist_modes[5]

        self.xyzfit_flag = False     # manual "phydist_contrast" option

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
            
            # d['rgb'] = torch.zeros([1], dtype=torch.float32, device=results['rgb'].device)

            o = results['opacity']+1e-10
            # encourage opacity to be either 0 or 1 to avoid floater
            if self.lambda_opacity > 0:
                d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

            # haoxiang_flag = True            
            # if haoxiang_flag:
            #     # --- for image with masked background (black) --- #
            #     d['mask'] = 100 * o[target['rgb'].sum(-1) == 0]
                

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
                if self.phydist_mode == "phydist_contrast" or self.xyzfit_flag == True:
                    sam_o = results['semantic'][batchsize//2:, :][valid_semantic_idx.squeeze(), :-3]
                    sam_x = results['semantic'][batchsize//2:, :][valid_semantic_idx.squeeze(), -3:]    # network output (x, y, z)
            else:
                valid_semantic_idx = target['sam'] > 0
                
                # haoxiang_flag = True            
                # if haoxiang_flag:   # only keep foreground points for supervision
                #     valid_semantic_idx = (target['sam'] > 0) & (torch.sum(target['rgb'], dim=1, keepdim=True) > 0)

                sam_t = target['sam'][valid_semantic_idx].long()
                sam_o = results['semantic'][valid_semantic_idx.squeeze(), :]
                if self.phydist_mode == "phydist_contrast" or self.xyzfit_flag == True:
                    sam_o = results['semantic'][valid_semantic_idx.squeeze(), :-3]
                    sam_x = results['semantic'][valid_semantic_idx.squeeze(), -3:]    # network output (x, y, z)
            
            rnum = sam_t.shape[0]   # ray number
            if self.normalize_sam:
                sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6)
            



            ## --- option 3: contructive clustering --- #
            cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
            cluster_ids = cluster_ids[cnums_all > self.min_pixnum]
            cnums = cnums_all[cnums_all > self.min_pixnum]
            cnum = cluster_ids.shape[0] # cluster number

            u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
            phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)

            if self.phydist_mode != "None":
                rays_o = results['rays_o']
                rays_d = results['rays_d']
                if hasattr(target, 'depth'):
                    depth = target['depth']
                else:
                    depth = results['depth'].detach()
                pts3d = rays_o + rays_d * depth[:, None]
                if ray_sampling_mixture:
                    pts3d = pts3d[batchsize//2:, :][valid_semantic_idx.squeeze()]
                else:
                    pts3d = pts3d[valid_semantic_idx.squeeze()]
                phycenter_list = torch.zeros([cnum, 3], dtype=torch.float32, device=sam_o.device)

            for i in range(cnum):
                cluster = sam_o[sam_t == cluster_ids[i], :]
                u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
                phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))

                if self.phydist_mode != "None":
                    depth_cluster = pts3d[sam_t == cluster_ids[i], :]
                    phycenter_list[i] = torch.mean(depth_cluster, dim=0)

            if self.patch_flag:
                accpatch = target['accpatch']


            # tau = 0.1; phi_list[:, 0] = tau    # option 1: constant temperature
            # phi_list = phi_list * (tau / phi_list.mean())     # option 2: (PCL) too small phi causes too large num in torch.exp().
            # phi_list = (phi_list - phi_list.min()) / (phi_list.max() - phi_list.min()) * 5 + 0.1   # scale to range [0.1, 5.1]
            phi_list = torch.clip(phi_list * 10, min=0.1, max=1.0)
            phi_list = phi_list.detach()
            
            ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

            if (self.phydist_mode == "order_loss") or (self.phydist_mode == "phydist_contrast") or (self.phydist_mode == "order_loss_supp") or self.xyzfit_flag == True:
                Distloss = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

            for i in range(cnum):
                cluster = sam_o[sam_t == cluster_ids[i], :]

                if self.phydist_mode == "None":

                    dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]
                    ProtoNCE += -torch.sum(torch.log(
                        dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                        ))

                else:
                    dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]


                    if self.phydist_mode == "ratio_regularization":
                        # --- option: 1  Distance Regularization --- #
                        # depth_cluster = pts3d[sam_t == cluster_ids[i], :]
                        # phydist = torch.norm(depth_cluster[:, None, :] - phycenter_list[None, :, :], dim=-1)    # (C_pix_num, Nc)
                        phydist = torch.norm(phycenter_list[:, :] - phycenter_list[[i], :], dim=-1)
                        phydist = (phydist - phydist.min()) / (phydist.max() - phydist.min()) * 99 + 1   # (1, 10)
                        
                        ProtoNCE += -torch.sum(torch.log(
                            dist[:, [i]] / ((dist[:, :] * phydist[None, :]).sum(dim=1, keepdim=True) + 1e-6)
                            ))
                    
                    elif self.phydist_mode == "patch-segment":
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
                                # max_loss = log.mean(1).sum()
                                # max_log = log.amax(1, True)
                            
                            # ProtoNCE += max_loss
                            # ProtoNCE += max_loss / levelnum
                            ProtoNCE += max_loss * self.sam_level_weight**(-l)     # --- level weighting --- #
                        

                        if self.xyzfit_flag:
                            # --- simple distance fitting --- #

                            depth_cluster = sam_x[sam_t == cluster_ids[i], :]
                            phydist = torch.norm(depth_cluster[:, :] - phycenter_list[[i], :], dim=-1)    # (C_pix_num, Nc)
                            distcontra_loss = phydist.mean()
                            
                            Distloss += distcontra_loss

                    
                    elif self.phydist_mode == "order_loss":
                        # --- option: 2.1  Distance Order loss --- #
                        phydist = torch.norm(phycenter_list[:, :] - phycenter_list[[i], :], dim=-1)
                        phydist = (phydist - phydist.min()) / (phydist.max() - phydist.min())   # (0, 1)
                        order = torch.sort(phydist, descending=False).indices
                        featdist = torch.matmul(u_list, u_list[i, :][:, None])
                        featdist_order = featdist[order]
                        order_weight = torch.arange(1, cnum+1, dtype=torch.float32, device=sam_o.device) * 2 - (cnum + 1)     # sum{r=1, r=n}(2r-n-1)*ar
                        order_weight = order_weight / torch.abs(order_weight).sum()
                        order_loss = order_weight @ featdist_order.squeeze()

                        Distloss += order_loss

                        ProtoNCE += -torch.sum(torch.log(
                            dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                            ))

                    elif self.phydist_mode == "order_loss_supp":    # order loss suppress version: 
                        # --- option: 2.2  Distance Order loss --- #
                        phydist = torch.norm(phycenter_list[:, :] - phycenter_list[[i], :], dim=-1)
                        phydist = (phydist - phydist.min()) / (phydist.max() - phydist.min())   # (0, 1)
                        order = torch.sort(phydist, descending=False).indices
                        
                        # u_list_norm = u_list / u_list.norm(dim=1, keepdim=True)
                        # featdist = torch.matmul(u_list_norm, u_list_norm[i, :][:, None])
                        featdist = torch.matmul(u_list, u_list[i, :][:, None])
                        featdist_order = featdist[order]

                        # temp_max = featdist_order[1]-featdist_order[0]
                        # for oi in range(1, cnum):
                        #     temp_max = max(temp_max, featdist_order[oi-1] - featdist_order[oi])
                        #     Distloss += temp_max

                        temp_max = featdist_order[1]
                        for oi in range(2, cnum):
                            if temp_max < featdist_order[oi]:
                                Distloss += - temp_max
                                break
                            else:
                                Distloss += - featdist_order[oi]
                            
                        # order_weight = torch.arange(1, cnum+1, dtype=torch.float32, device=sam_o.device) * 2 - (cnum + 1)     # sum{r=1, r=n}(2r-n-1)*ar
                        # order_weight = order_weight / torch.abs(order_weight).sum()
                        # order_loss = order_weight @ featdist_order.squeeze()
                        # Distloss += order_loss

                        ProtoNCE += -torch.sum(torch.log(
                            dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                            ))

                    elif self.phydist_mode == "phydist_contrast":
                        depth_cluster = sam_x[sam_t == cluster_ids[i], :]

                        # --- option1: contrastive - not need to contrast actually --- #
                        # phydist = torch.norm(depth_cluster[:, None, :] - phycenter_list[None, :, :], dim=-1)    # (C_pix_num, Nc)
                        # distcontra_loss = torch.sum(phydist[:, [i]] / (phydist[:, :].sum(dim=1, keepdim=True) + 1e-6))
                        
                        # --- option2: simple distance fitting --- #
                        phydist = torch.norm(depth_cluster[:, :] - phycenter_list[[i], :], dim=-1)    # (C_pix_num, Nc)
                        distcontra_loss = phydist.mean()
                        
                        Distloss += distcontra_loss

                        ProtoNCE += -torch.sum(torch.log(
                            dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
                            ))
                    else:
                        print("Unrecognized Phydist option, exit now!")
                        exit(0)

            d['semantic_render'] = self.lambda_sam * ProtoNCE

            if self.phydist_mode == "order_loss":
                d['sem_phydist'] = 0.2 * Distloss
            if self.phydist_mode == "order_loss_supp":
                d['sem_phydist'] = 0.2 * Distloss
            elif self.phydist_mode == "phydist_contrast" or self.xyzfit_flag == True:
                d['sem_phydist'] = 0.1 * Distloss

            if self.sam_norm_loss_flag:
                sam_norm_loss = ((torch.norm(sam_o, dim=-1, keepdim=True) - 1.0) ** 2).mean()
                d['semantic_render'] += 100 * sam_norm_loss



            # --- debug --- #
            debug = False
            if debug:
                pix_t = target['pix_idxs'][batchsize//2:][valid_semantic_idx.squeeze()]
                ii = 1
                print((pix_t[sam_t == cluster_ids[ii]] % 400).min(), (pix_t[sam_t == cluster_ids[ii]] % 400).max(), \
                      (pix_t[sam_t == cluster_ids[ii]] // 400).min(), (pix_t[sam_t == cluster_ids[ii]] // 400).max())



        return d
