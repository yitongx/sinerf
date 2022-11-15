'''
For trajectory visualization
'''
import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.join(sys.path[0], '../..'))

import open3d as o3d
from utils.vis_cam_traj import draw_camera_frustum_geometry, get_camera_frustum_geometry_nparray

import torch
import numpy as np

from dataloader.with_colmap import DataLoaderWithCOLMAP
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.align_traj import align_ate_c2b_use_a2b, pts_dist_max
from utils.comp_ate import compute_ate
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data/LLFF/',
                        help='folder contains various scenes')
    parser.add_argument('--scene_name', type=str, default='flower')
    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--fx_only', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)
    parser.add_argument('--init_pose_colmap', default=False, type=bool,
                        help='set this to True if the nerfmm model is trained from COLMAP init.')
    parser.add_argument('--init_focal_colmap', default=False, type=bool,
                        help='set this to True if the nerfmm model is trained from COLMAP init.')

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--ATE_align', type=bool, default=True)
    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')
    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def main(args):
    my_devices = torch.device('cpu')

    '''Create Folders'''
    pose_out_dir = Path(os.path.join(args.ckpt_dir, 'pose_out'))
    pose_out_dir.mkdir(parents=True, exist_ok=True)

    '''Get COLMAP poses'''
    scene_train = DataLoaderWithCOLMAP(base_dir=args.base_dir,
                                       scene_name=args.scene_name,
                                       data_type='train',
                                       res_ratio=args.resize_ratio,
                                       num_img_to_load=args.train_img_num,
                                       skip=args.train_skip,
                                       use_ndc=True,
                                       load_img=False)

    # scale colmap poses to unit sphere
    ts_colmap = scene_train.c2ws[:, :3, 3]  # (N, 3)
    scene_train.c2ws[:, :3, 3] /= pts_dist_max(ts_colmap)
    scene_train.c2ws[:, :3, 3] *= 2.0

    '''Load scene meta'''
    H, W = scene_train.H, scene_train.W
    colmap_focal = scene_train.focal

    print('Intrinsic: H: {0:4d}, W: {1:4d}, COLMAP focal {2:.2f}.'.format(H, W, colmap_focal))

    '''Model Loading'''
    if args.init_focal_colmap:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=colmap_focal)
    else:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order)
        # only load learned focal if we do not init with colmap focal
        focal_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)
    fxfy = focal_net(0)
    print('COLMAP focal: {0:.2f}, learned fx: {1:.2f}, fy: {2:.2f}'.format(colmap_focal, fxfy[0].item(), fxfy[1].item()))

    if args.init_pose_colmap:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, scene_train.c2ws)
    else:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    pose_param_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), pose_param_net, map_location=my_devices)

    '''Get all poses in (N, 4, 4)'''
    c2ws_est = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])  # (N, 4, 4)
    c2ws_cmp = scene_train.c2ws  # (N, 4, 4)

    # scale estimated poses to unit sphere
    ts_est = c2ws_est[:, :3, 3]  # (N, 3)
    c2ws_est[:, :3, 3] /= pts_dist_max(ts_est)
    c2ws_est[:, :3, 3] *= 2.0

    '''Define camera frustums'''
    frustum_length = 0.05
    est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
    cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255

    '''Align est traj to colmap traj'''
    c2ws_est_to_draw_align2cmp = c2ws_est.clone()
    if args.ATE_align:  # Align learned poses to colmap poses
        c2ws_est_aligned = align_ate_c2b_use_a2b(c2ws_est, c2ws_cmp)  # (N, 4, 4)
        c2ws_est_to_draw_align2cmp = c2ws_est_aligned

        # compute ate
        stats_tran_est, stats_rot_est, _ = compute_ate(c2ws_est_aligned, c2ws_cmp, align_a2b=None)
        print('From est to colmap: tran err {0:.3f}, rot err {1:.2f}'.format(stats_tran_est['mean'],
                                                                             stats_rot_est['mean']))

    # frustum_est_points_list, frustum_est_lines_list, _ = get_camera_frustum_geometry_nparray_list(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W, 
    #                                                             fxfy[0], fxfy[1], frustum_length, est_traj_color)
    # frustum_colmap_points_list, frustum_colmap_lines_list, _ = get_camera_frustum_geometry_nparray_list(c2ws_cmp.cpu().numpy(), H, W, 
    #                                                                colmap_focal, colmap_focal, frustum_length, cmp_traj_color)
    # print("len(frustum_est_list.points):", len(frustum_est_points_list), frustum_est_points_list[0].shape)  # N x (5, 3)
    # print("len(frustum_est_list.lines):", len(frustum_est_lines_list), frustum_est_lines_list[0].shape) # N x (8, 2)
    
    frustum_est_points_np, frustum_est_lines_np, _ = get_camera_frustum_geometry_nparray(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W, fxfy[0], fxfy[1], frustum_length, est_traj_color)
    frustum_colmap_points_np, frustum_colmap_lines_np, _ = get_camera_frustum_geometry_nparray(c2ws_cmp.cpu().numpy(), H, W, colmap_focal, colmap_focal, frustum_length, cmp_traj_color)

    print("scene_train.N_imgs: ", scene_train.N_imgs)  
    print("frustum_est_points_np:", frustum_est_points_np.shape)
    print("frustum_est_lines_np:", frustum_est_lines_np.shape)

    '''pyplot draw'''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    '''Only get origin points'''
    plt_est_origins = np.zeros((scene_train.N_imgs, 3))
    plt_colmap_origins = np.zeros((scene_train.N_imgs, 3))
    for i in range(scene_train.N_imgs):
        plt_est_origins[i] = frustum_est_points_np[5*i]
        plt_colmap_origins[i] = frustum_colmap_points_np[5*i]
        
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(plt_colmap_origins[:, 0], plt_colmap_origins[:, 1], 'm-^')
    ax.plot(plt_est_origins[:, 0], plt_est_origins[:, 1], 'b-^')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.0, 1.0])
    ax.legend(['COLMAP', 'Ours'], loc='upper left')
    ax.set_title("{}".format(args.scene_name[0].upper() + args.scene_name[1:].lower()))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_zlabel('z')
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
