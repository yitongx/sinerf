from re import A
import torch
from torch.distributions import Categorical

def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def get_ndc_rays_fxfy(H, W, fxfy, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * fxfy[0])) * ox_oz
    o1 = -1. / (H / (2. * fxfy[1])) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * fxfy[0])) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * fxfy[1])) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def volume_sampling(c2w, ray_dir_cam, t_vals, near, far, perturb_t):
    """
    :param c2w:             (4, 4)                  camera pose
    :param ray_dir_cam:     (H, W, 3)               ray directions in the camera coordinate
    :param t_vals:          (N_sample, )            sample depth in a ray. No lindisp in default nerfmm.
    :param perturb_t:       True/False              whether add noise to t
    """
    ray_H, ray_W = ray_dir_cam.shape[0], ray_dir_cam.shape[1]
    N_sam = t_vals.shape[0]

    # transform rays from camera coordinate to world coordinate
    ray_dir_world = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    ray_ori_world = c2w[:3, 3]  # the translation vector (3, )

    # this perturb only works if we sample depth linearly, not the disparity.
    if perturb_t:
        # add some noise to each each z_val
        t_noise = torch.rand((ray_H, ray_W, N_sam), device=c2w.device, dtype=torch.float32)  # (H, W, N_sam)
        t_noise = t_noise * (far - near) / N_sam
        t_vals_noisy = t_vals.view(1, 1, N_sam) + t_noise  # (H, W, N_sam)
    else:
        t_vals_noisy = t_vals.view(1, 1, N_sam).expand(ray_H, ray_W, N_sam)

    # Get sample position in the world (1, 1, 1, 3) + (H, W, 1, 3) * (H, W, N_sam, 1) -> (H, W, N_sample, 3)
    sample_pos = ray_ori_world.view(1, 1, 1, 3) + ray_dir_world.unsqueeze(2) * t_vals_noisy.unsqueeze(3)

    return sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy  # (H, W, N_sample, 3), (3, ), (H, W, 3), (H, W, N_sam)


def volume_sampling_ndc(c2w, ray_dir_cam, t_vals, near, far, H, W, focal, perturb_t):
    """
    :param c2w:             (3/4, 4)                camera pose
    :param ray_dir_cam:     (H, W, 3)               ray directions in the camera coordinate
    :param focal:           a float or a (2,) torch tensor for focal.
    :param t_vals:          (N_sample, )            sample depth in a ray
    :param perturb_t:       True/False              whether add noise to t
    """
    ray_H, ray_W = ray_dir_cam.shape[0], ray_dir_cam.shape[1]
    N_sam = t_vals.shape[0]

    # transform rays from camera coordinate to world coordinate
    ray_dir_world = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    ray_ori_world = c2w[:3, 3]  # the translation vector (3, )

    ray_dir_world = ray_dir_world.reshape(-1, 3)  # (H, W, 3) -> (H*W, 3)
    ray_ori_world = ray_ori_world.view(1, 3).expand_as(ray_dir_world)  # (3, ) -> (1, 3) -> (H*W, 3)
    if isinstance(focal, float):
        ray_ori_world, ray_dir_world = get_ndc_rays(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)
    else:  # if focal is a tensor contains fxfy
        ray_ori_world, ray_dir_world = get_ndc_rays_fxfy(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)
    ray_dir_world = ray_dir_world.reshape(ray_H, ray_W, 3)  # (H, W, 3)
    ray_ori_world = ray_ori_world.reshape(ray_H, ray_W, 3)  # (H, W, 3)

    # this perturb only works if we sample depth linearly, not the disparity.
    if perturb_t:
        # add some noise to each each z_val
        t_noise = torch.rand((ray_H, ray_W, N_sam), device=c2w.device, dtype=torch.float32)  # (H, W, N_sam)
        t_noise = t_noise * (far - near) / N_sam
        t_vals_noisy = t_vals.view(1, 1, N_sam) + t_noise  # (H, W, N_sam)
    else:
        t_vals_noisy = t_vals.view(1, 1, N_sam).expand(ray_H, ray_W, N_sam)

    # Get sample position in the world (H, W, 1, 3) + (H, W, 1, 3) * (H, W, N_sam, 1) -> (H, W, N_sample, 3)
    sample_pos = ray_ori_world.unsqueeze(2) + ray_dir_world.unsqueeze(2) * t_vals_noisy.unsqueeze(3)

    return sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy  # (H, W, N_sample, 3), (3, ), (H, W, 3), (H, W, N_sam)

def volume_sampling_ndc_zvals(c2w, ray_dir_cam, t_vals, near, far, H, W, focal, perturb_t):
    """
    For hierachical sampling z_vals.
    :param c2w:             (3/4, 4)                camera pose
    :param ray_dir_cam:     (H, W, 3)               ray directions in the camera coordinate
    :param focal:           a float or a (2,) torch tensor for focal.
    :param t_vals:          (H, W, N_sample)            sample depth in a ray
    :param perturb_t:       True/False              whether add noise to t
    """
    ray_H, ray_W = ray_dir_cam.shape[0], ray_dir_cam.shape[1]
    N_sam = t_vals.shape[0]

    # transform rays from camera coordinate to world coordinate
    ray_dir_world = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    ray_ori_world = c2w[:3, 3]  # the translation vector (3, )

    ray_dir_world = ray_dir_world.reshape(-1, 3)  # (H, W, 3) -> (H*W, 3)
    ray_ori_world = ray_ori_world.view(1, 3).expand_as(ray_dir_world)  # (3, ) -> (1, 3) -> (H*W, 3)
    if isinstance(focal, float):
        ray_ori_world, ray_dir_world = get_ndc_rays(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)
    else:  # if focal is a tensor contains fxfy
        ray_ori_world, ray_dir_world = get_ndc_rays_fxfy(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)
    ray_dir_world = ray_dir_world.reshape(ray_H, ray_W, 3)  # (H, W, 3)
    ray_ori_world = ray_ori_world.reshape(ray_H, ray_W, 3)  # (H, W, 3)

    # this perturb only works if we sample depth linearly, not the disparity.
    if perturb_t:
        # add some noise to each each z_val
        t_noise = torch.rand((ray_H, ray_W, N_sam), device=c2w.device, dtype=torch.float32)  # (H, W, N_sam)
        t_noise = t_noise * (far - near) / N_sam
        t_vals_noisy = t_vals + t_noise  # (H, W, N_sam)
    else:
        t_vals_noisy = t_vals

    # Get sample position in the world (H, W, 1, 3) + (H, W, 1, 3) * (H, W, N_sam, 1) -> (H, W, N_sample, 3)
    sample_pos = ray_ori_world.unsqueeze(2) + ray_dir_world.unsqueeze(2) * t_vals_noisy.unsqueeze(3)

    return sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy  # (H, W, N_sample, 3), (3, ), (H, W, 3), (H, W, N_sam)


def volume_rendering(rgb_density, t_vals, sigma_noise_std, rgb_act_fn):
    """
    :param rgb_density:     (H, W, N_sample, 4)     network output
    :param t_vals:          (H, W, N_sample)        compute the distance between each sample points
    :param sigma_noise_std: A scalar                add some noise to the density output, this is helpful to reduce
                                                    floating artifacts according to official repo, but they set it to
                                                    zero in their implementation.
    :param rgb_act_fn:      relu()                  apply an active fn to the raw rgb output to get actual rgb
    :return:                (H, W, 3)               rendered rgb image
                            (H, W, N_sample)        weights at each sample position
    """
    ray_H, ray_W, num_sample = t_vals.shape[0], t_vals.shape[1], t_vals.shape[2]

    rgb = rgb_act_fn(rgb_density[:, :, :, :3])  # (H, W, N_sample, 3)
    sigma_a = rgb_density[:, :, :, 3]  # (H, W, N_sample)

    if sigma_noise_std > 0.0:
        sigma_noise = torch.randn_like(sigma_a) * sigma_noise_std
        sigma_a = sigma_a + sigma_noise  # (H, W, N_sample)

    ### Default to be relu
    sigma_a = sigma_a.relu()  # (H, W, N_sample)
    # sigma_a = torch.log(1 + torch.exp(sigma_a))   # (H, W, N_sample)

    # Compute distances between samples.
    # 1. compute the distances among first (N-1) samples
    # 2. the distance between the LAST sample and infinite far is default to be 1e10
    # 3. Try set to 1e2 as in NeRF-W.
    dists = t_vals[:, :, 1:] - t_vals[:, :, :-1]  # (H, W, N_sample-1)
    dist_far = torch.empty(size=(ray_H, ray_W, 1), dtype=torch.float32, device=dists.device).fill_(1e10)  # (H, W, 1)
    # dist_far = torch.empty(size=(ray_H, ray_W, 1), dtype=torch.float32, device=dists.device).fill_(1e2)  # (H, W, 1)
    dists = torch.cat([dists, dist_far], dim=2)  # (H, W, N_sample)

    ### opacity alpha
    alpha = 1 - torch.exp(-1.0 * sigma_a * dists)  # (H, W, N_sample)

    # 1. We expand the exp(a+b) to exp(a) * exp(b) for the accumulated transmittance computing.
    # 2. For the space at the boundary far to camera, the alpha is constant 1.0 and the transmittance at the far boundary
    # is useless. For the space at the boundary near to camera, we manually set the transmittance to 1.0, which means
    # 100% transparent. The torch.roll() operation simply discards the transmittance at the far boundary.
    acc_transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=2)  # (H, W, N_sample)
    acc_transmittance = torch.roll(acc_transmittance, shifts=1, dims=2)  # (H, W, N_sample)
    acc_transmittance[:, :, 0] = 1.0  # (H, W, N_sample)

    weight = acc_transmittance * alpha  # (H, W, N_sample)

    rgb_rendered = torch.sum(weight.unsqueeze(3) * rgb, dim=2)  # (H, W, N_sample, 1) * (H, W, N_sample, 3) = (H, W, N_sample, 3) -> (H, W, 3)
    '''
    Use normalized weight for depth map
    '''
    normalized_weight = weight / (torch.sum(weight, dim=-1, keepdim=True) + 1e-10)
    depth_map = torch.sum(normalized_weight * t_vals, dim=2)  # (H, W)
    acc_map = torch.sum(normalized_weight, dim=2)  ### Accumulated opacity along each ray

    # depth_map = torch.sum(weight * t_vals, dim=2)  # (H, W)
    # acc_map = torch.sum(weight, dim=2)  ### Accumulated opacity along each ray, (H, W)

    # Calculate weights sparsity loss
    mask = weight.sum(-1) > 0.5  # (H, W)
    entropy = Categorical(probs = weight + 1e-5).entropy()  # (H, W)
    sparsity_loss = entropy * mask  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'weight': weight,  # (H, W, N_sample)
        'depth_map': depth_map,  # (H, W)
        'acc_map': acc_map, # (H, W)
        'sparsity_loss': sparsity_loss  # (H, W)
    }
    return result

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Inverse transform sampling for Hierachical sampling.
    :param bins: "z_vals_mid", midpoints of uniform sampling intervals..
    :param weights: Weights assigned to each sampled color. Exclude head and tail ones.
    :param N_samples: Extra amount of fine sampling points.
    :param det:
    :param pytest:
    :return:
    """
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # [batch, len(bins)]
    
    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])    ### [batch, Ns]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.to(cdf.device).contiguous()
    inds = torch.searchsorted(cdf, u, right=True)   ### [batch, Ns]
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # [batch, N_samples, 2]
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]   ### [batch, Ns, len(bins)]
    cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), -1, inds_g) ### [batch, N_samples, 2]
    bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), -1, inds_g)
    
    denom = (cdf_g[...,1] - cdf_g[...,0])   ### [batch, Ns]
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0]) ### accurate sampling within [below, above] interval

    return samples


# if __name__ == '__main__':
#     H = 756
#     W = 1004
#     n_sample = 64
#     n_sample_importance = 61
#     t_vals = torch.linspace(0, 1, n_sample)  # (N_sample,) sample position
#     weights = torch.randn(H, W, n_sample)
#     weights = weights[..., 1:-1]   # （H, W, N-2)           
#     bins = 0.5 * (t_vals[None, None, 1:] - t_vals[None, None, :-1])
#     bins = bins.expand(H, W, bins.shape[-1])   # （H, W, N-1)  
#     t_vals_importance = sample_pdf(bins, weights, n_sample_importance, True)
#     t_vals_importance = t_vals_importance.detach()
#     print(t_vals_importance.shape)  # (H, W, n_sample_importance)