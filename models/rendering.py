import torch
from .custom_functions import RayAABBIntersector, RayMarcher, VolumeRenderer, VolumeRenderer_sam
from einops import rearrange
import vren

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    semantic_flag = kwargs.get('semantic_flag', False)
    semantic_dim = kwargs.get('semantic_dim', 4)
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    if semantic_flag is True:
        semantic = torch.zeros(N_rays, semantic_dim, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    alive_indices_delay = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)

        if semantic_flag:
            semantics = torch.zeros(len(xyzs), semantic_dim, device=device)
            sigmas[valid_mask], _rgbs, _semantics = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        else:
            sigmas[valid_mask], _rgbs = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)

        rgbs[valid_mask] = _rgbs.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)


        if semantic_flag:
            semantics[valid_mask] = _semantics.float()
            semantics = rearrange(semantics, '(n1 n2) c -> n1 n2 c', n2=N_samples)
            vren.composite_sam_test_fw(
                sigmas, rgbs, semantics, deltas, ts,
                hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4), semantic_dim,
                N_eff_samples, opacity, depth, rgb, semantic)
            alive_indices = alive_indices[alive_indices>=0] # remove converged rays

            results['semantic'] = semantic
        else:
            vren.composite_test_fw(         # rays have been removed from alive_indices, but new ray-marching information still remain in sigmas, rgbs, deltas, and ts, which causes misplacement in func vren.composite_test_fw()
                sigmas, rgbs, deltas, ts,
                hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
                N_eff_samples, opacity, depth, rgb)
            
            alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['xyz'] = rays_o + rays_d * (depth / opacity.clamp(1e-3))[:, None]
    results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays

    if exp_step_factor==0 or kwargs.get('white_bg', False):
        rgb_bg = torch.ones(3, device=device)
    else:
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')


    bg_color = kwargs.get('bg_color', 0)
    if bg_color == 1:
        rgb_bg = torch.ones(3, device=rays_o.device)
    else:
        rgb_bg = torch.zeros(3, device=rays_o.device)

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    semantic_flag = kwargs.get('semantic_flag', False)
    semantic_dim = kwargs.get('semantic_dim', 4)
    semantic_only = kwargs.get('semantic_only', False)
    results = {}

    (rays_a, xyzs, dirs,
    results['deltas'], results['ts'], results['rm_samples']) = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)

    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

    if semantic_flag:
        sigmas, rgbs, semantics = model(xyzs, dirs, **kwargs)
        rgbs = rgbs.contiguous()

        (results['vr_samples'], results['opacity'], results['depth'], _, results['semantic'], _) = \
            VolumeRenderer_sam.apply(sigmas.detach(), rgbs.detach(), semantics.contiguous(), results['deltas'], results['ts'], rays_a, kwargs.get('T_threshold', 1e-4), semantic_dim)

        if not semantic_only:
            (_, _, _, results['rgb'], results['ws']) = \
                VolumeRenderer.apply(sigmas, rgbs, results['deltas'], results['ts'], rays_a, kwargs.get('T_threshold', 1e-4))
        results['semantics_pts'] = semantics
    else:
        sigmas, rgbs = model(xyzs, dirs, **kwargs)
        (results['vr_samples'], results['opacity'],
        results['depth'], results['rgb'], results['ws']) = \
            VolumeRenderer.apply(sigmas, rgbs, results['deltas'], results['ts'],
                                 rays_a, kwargs.get('T_threshold', 1e-4))

    if not semantic_only:
        results['rays_a'] = rays_a

        bg_color = kwargs.get('bg_color', 0)
        if bg_color == 0:
            rgb_bg = torch.zeros(3, device=rays_o.device)
        elif bg_color == 1:
            rgb_bg = torch.ones(3, device=rays_o.device)
        elif bg_color == 2:
            rgb_bg = torch.rand_like(results['rgb'])

        results['rgb'] = results['rgb'] + \
                        rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

    return results
