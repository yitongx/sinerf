import os
from turtle import down
import cv2
from cv2 import blur

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio

from utils.comp_ray_dir import comp_ray_dir_cam
from utils.pose_utils import center_poses
from utils.lie_group_helper import convert3x4_4x4

import torchvision.transforms as transforms

def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB


def load_imgs(image_dir, img_ids, new_h, new_w, interest_region_flag=False):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]
    img_list = []
    interest_indices_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
        img_resize = cv2.resize(img, (new_w, new_h))
        cords_list = find_POI(img_resize)
        interest_indices = get_interest_region_cords(cords_list, new_h, new_w)  # (N, 2) np.ndarray
        interest_indices_list.append(interest_indices)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    img_list = resize_imgs(img_list, new_h, new_w)
    
    if interest_region_flag == True:
        return img_list, img_names, interest_indices_list
    return img_list, img_names


def load_split(scene_dir, img_dir, data_type, num_img_to_load, skip, c2ws,
               H, W, load_img, interest_region_flag=False):
    # load pre-splitted train/val ids
    img_ids = np.loadtxt(os.path.join(scene_dir, data_type + '_ids.txt'), dtype=np.int32, ndmin=1)
    if num_img_to_load == -1:
        img_ids = img_ids[::skip]
        print('Loading all available {0:6d} images'.format(len(img_ids)))
    elif num_img_to_load > len(img_ids):
        print('Required {0:4d} images but only {1:4d} images available. '
              'Exit'.format(num_img_to_load, len(img_ids)))
        exit()
    else:
        img_ids = img_ids[:num_img_to_load:skip]

    N_imgs = img_ids.shape[0]

    # use img_ids to select camera poses
    c2ws = c2ws[img_ids]  # (N, 3, 4)

    # load images
    interest_indices = None
    if load_img:
        if not interest_region_flag:
            imgs, img_names = load_imgs(img_dir, img_ids, H, W, False)  # (N, H, W, 3) torch.float32
        else:
            imgs, img_names, interest_indices = load_imgs(img_dir, img_ids, H, W, True)
    else:
        imgs, img_names = None, None

    result = {
        'c2ws': c2ws,  # (N, 3, 4) np.float32
        'imgs': imgs,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'img_ids': img_ids,  # (N, ) np.int
        'interest_indices': interest_indices,   # (N, 2)
    }
    return result


def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    poses_bounds = np.load(os.path.join(in_dir, 'poses_bounds.npy'))  # (N_images, 17)

    c2ws = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = c2ws[0, :, -1]

    # correct c2ws: original c2ws has rotation in form "down right back", change to "right up back".
    # See https://github.com/bmild/nerf/issues/34
    c2ws = np.concatenate([c2ws[..., 1:2], -c2ws[..., :1], c2ws[..., 2:4]], -1)

    # (N_images, 3, 4), (4, 4)
    c2ws, pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws

    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        c2ws[..., 3] /= scale_factor    # scaling H, W, f accordingly
    
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    results = {
        'c2ws': c2ws,       # (N, 4, 4) np
        'bounds': bounds,   # (N_images, 2) np
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': focal,     # scalar
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results


class DataLoaderWithCOLMAP:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, data_type, res_ratio, num_img_to_load, skip, use_ndc, load_img=True, use_ROI=False):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img
        self.use_ROI = use_ROI
        
        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(self.scene_dir, self.use_ndc)
        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = float(meta['focal'])
        self.total_N_imgs = self.c2ws.shape[0]

        if self.res_ratio > 1:
            self.H = self.H // self.res_ratio
            self.W = self.W // self.res_ratio
            self.focal /= self.res_ratio

        self.near = 0.0
        self.far = 1.0

        '''Load train/val split'''
        split_results = load_split(self.scene_dir, self.img_dir, self.data_type, self.num_img_to_load,
                                   self.skip, self.c2ws, self.H, self.W, self.load_img, self.use_ROI)
        self.c2ws = split_results['c2ws']  # (N, 4, 4) np.float32
        self.imgs = split_results['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = split_results['img_names']  # (N, )
        self.N_imgs = split_results['N_imgs']
        self.img_ids = split_results['img_ids']  # (N, ) np.int
        self.interest_indices = split_results['interest_indices']   # (N, 2) np.int

        # generate cam ray dir.
        self.ray_dir_cam = comp_ray_dir_cam(self.H, self.W, self.focal)  # (H, W, 3) torch.float32

        # convert np to torch.
        self.c2ws = torch.from_numpy(self.c2ws).float()  # (N, 4, 4) torch.float32
        self.ray_dir_cam = self.ray_dir_cam.float()  # (H, W, 3) torch.float32


def downsample_and_resize_imgs(imgs, down_h, down_w, new_h, new_w, blur_flag=False):
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(down_h, down_w), mode='bilinear')  # (N, 3, down_h, down_w)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='nearest')  # (N, 3, new_H, new_W)
    if blur_flag:
        ks = new_h//down_h+1
        if ks % 2 == 0: ks += 1
        trans = transforms.GaussianBlur(kernel_size=(ks, ks), sigma=(1, 1))   ### trans must be applied to NCHW tensor
        imgs = trans(imgs)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)
    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB

def downsample_and_resize_load_imgs(image_dir, img_ids, down_h, down_w, new_h, new_w, blur_flag):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255.  # (N, H, W, 3) torch.float32
    img_list = downsample_and_resize_imgs(img_list, down_h, down_w, new_h, new_w, blur_flag)
    return img_list, img_names

def downsample_and_resize_load_split(scene_dir, 
                                     img_dir, 
                                     data_type, 
                                     num_img_to_load, 
                                     skip, 
                                     c2ws, 
                                     down_H, 
                                     down_W,
                                     new_H,
                                     new_W,
                                     load_img):
    # load pre-splitted train/val ids
    img_ids = np.loadtxt(os.path.join(scene_dir, data_type + '_ids.txt'), dtype=np.int32, ndmin=1)
    if num_img_to_load == -1:
        img_ids = img_ids[::skip]
        print('Loading all available {0:6d} images'.format(len(img_ids)))
    elif num_img_to_load > len(img_ids):
        print('Required {0:4d} images but only {1:4d} images available. '
              'Exit'.format(num_img_to_load, len(img_ids)))
        exit()
    else:
        img_ids = img_ids[:num_img_to_load:skip]

    N_imgs = img_ids.shape[0]

    # use img_ids to select camera poses
    c2ws = c2ws[img_ids]  # (N, 3, 4)

    # load images
    blur_flag = False if (down_H == new_H and down_W == new_W) else True
    if load_img:
        imgs, img_names = downsample_and_resize_load_imgs(img_dir, img_ids, down_H, down_W, new_H, new_W, blur_flag)  # (N, H, W, 3) torch.float32
    else:
        imgs, img_names = None, None

    result = {
        'c2ws': c2ws,  # (N, 3, 4) np.float32
        'imgs': imgs,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'img_ids': img_ids,  # (N, ) np.int
    }
    return result


class DataLoaderWithCOLMAP_multires:
    """
    Down sample and then resize to the original size (raw res // 4)
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, 
                 base_dir, 
                 scene_name, 
                 data_type, 
                 res_ratio, 
                 num_img_to_load, 
                 skip, 
                 use_ndc, 
                 load_img=True):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img

        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(self.scene_dir, self.use_ndc)
        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = float(meta['focal'])
        self.total_N_imgs = self.c2ws.shape[0]

        ### for downsize image
        if self.res_ratio > 1:
            self.down_H = self.H // self.res_ratio
            self.down_W = self.W // self.res_ratio
            # self.focal /= self.res_ratio

        ### final image resize default=4
        self.default_final_ratio = 4
        self.new_H = self.H // self.default_final_ratio
        self.new_W = self.W // self.default_final_ratio
        self.new_focal = self.focal / self.default_final_ratio
        self.H = self.H // self.default_final_ratio
        self.W = self.W // self.default_final_ratio
        self.focal /= self.default_final_ratio

        self.near = 0.0
        self.far = 1.0

        '''Load train/val split'''
        # split_results = load_split(self.scene_dir, self.img_dir, self.data_type, self.num_img_to_load,
        #                            self.skip, self.c2ws, self.H, self.W, self.load_img)
        split_results = downsample_and_resize_load_split(self.scene_dir, self.img_dir, self.data_type, self.num_img_to_load, 
                                                         self.skip, self.c2ws, self.down_H, self.down_W, self.new_H, self.new_W, self.load_img)
        self.c2ws = split_results['c2ws']  # (N, 4, 4) np.float32
        self.imgs = split_results['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = split_results['img_names']  # (N, )
        self.N_imgs = split_results['N_imgs']
        self.img_ids = split_results['img_ids']  # (N, ) np.int

        # generate cam ray dir. Following resize=4 settings
        self.ray_dir_cam = comp_ray_dir_cam(self.new_H, self.new_W, self.new_focal)  # (H, W, 3) torch.float32

        # convert np to torch.
        self.c2ws = torch.from_numpy(self.c2ws).float()  # (N, 4, 4) torch.float32
        self.ray_dir_cam = self.ray_dir_cam.float()  # (H, W, 3) torch.float32


def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        cv2.imwrite("./sift.png", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates

def get_interest_region_cords(cords_list, H, W):
    interest_regions = np.zeros((H, W), dtype=np.uint8)
    interest_regions[cords_list[:,1], cords_list[:,0]] = 1
    interest_regions = cv2.dilate(interest_regions, np.ones((5, 5), np.uint8), iterations=1)
    interest_regions = np.array(interest_regions, dtype=bool)
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H)), -1), dtype=int)
    interest_indices = coords[interest_regions]

    return interest_indices # (N, 2)

if __name__ == '__main__':
    scene_name = 'LLFF/fern'
    use_ndc = True
    scene = DataLoaderWithCOLMAP(base_dir='/your/data/path',
                                 scene_name=scene_name,
                                 data_type='train',
                                 res_ratio=8,
                                 num_img_to_load=-1,
                                 skip=1,
                                 use_ndc=use_ndc)
