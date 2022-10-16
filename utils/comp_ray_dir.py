from this import d
import torch


def comp_ray_dir_cam(H, W, focal):
    """Compute ray directions in the camera coordinate, which only depends on intrinsics.
    This could be further transformed to world coordinate later, using camera poses.
    :return: (H, W, 3) torch.float32
    """
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                          torch.arange(W, dtype=torch.float32))  # (H, W)

    # Use OpenGL coordinate in 3D:
    #   x points to right
    #   y points to up
    #   z points to backward
    #
    # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
    dirs_x = (x - 0.5*W + 0.5) / focal  # (H, W)
    dirs_y = -(y - 0.5*H + 0.5) / focal  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32)  # (H, W)
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    return rays_dir


def comp_ray_dir_cam_fxfy(H, W, fx, fy):
    """Compute ray directions in the camera coordinate, which only depends on intrinsics.
    This could be further transformed to world coordinate later, using camera poses.
    :return: (H, W, 3) torch.float32
    """
    ## pixel_y down, pixel_x right
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=fx.device),
                          torch.arange(W, dtype=torch.float32, device=fx.device))  # (H, W)

    # Use OpenGL coordinate in 3D:
    #   x points to right
    #   y points to up
    #   z points to backward
    #
    # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
    dirs_x = (x - 0.5*W + 0.5) / fx  # (H, W)
    dirs_y = -(y - 0.5*H + 0.5) / fy  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32, device=fx.device)  # (H, W)
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    return rays_dir


def comp_ray_dir_cam_radii_fxfy(H, W, fx, fy):
    """
    For mip encoding
    Compute ray directions in the camera coordinate, which only depends on intrinsics.
    This could be further transformed to world coordinate later, using camera poses.
    :return: 
    ray_dir (H, W, 3) torch.float32
    radii (H, W, 1) 
    """
    ## pixel_y down, pixel_x right
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=fx.device),
                          torch.arange(W, dtype=torch.float32, device=fx.device))  # (H, W)

    # Use OpenGL coordinate in 3D:
    #   x points to right
    #   y points to up
    #   z points to backward
    #
    # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
    dirs_x = (x - 0.5*W + 0.5) / fx  # (H, W)
    dirs_y = -(y - 0.5*H + 0.5) / fy  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32, device=fx.device)  # (H, W)
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    rays_dir = rays_dir / rays_dir.norm(dim=2, keepdim=True)

    ### compute radii
    dx = torch.sqrt(torch.sum((rays_dir[:, :-1, :] - rays_dir[:, 1:, :]) ** 2, dim=2))
    dx = torch.cat([dx, dx[:, -2:-1]], dim=1)
    dy = torch.sqrt(torch.sum((rays_dir[:-1, :, :] - rays_dir[1:, :, :]) ** 2, dim=2))
    dy = torch.cat([dy, dy[-2:-1, :]], dim=0)
    radii = (0.5 * (dx + dy))[..., None] * 2 / torch.sqrt(torch.tensor(12.0))   # (H, W, 1)
    
    return rays_dir, radii