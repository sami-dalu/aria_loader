from abc import ABC, abstractmethod
from copy import deepcopy

import cv2
import numpy as np
import trimesh
from loguru import logger

from objslam.utils.common import pathlib_file, get_all_files_with_suffix
from objslam.utils.constants import StringName
from objslam.utils.rand import FixedSeed
from objslam.utils.np_trans import apply_SE3
from objslam.utils.vision_3d import render_meshes_to_img
from objslam.utils.image import scale_cam_mat


class BaseVideo(ABC):
    GT_DEPTH_DIR = 'rendered_gt_depth'
    GT_DEPTH_PREFIX = None
    GT_DEPTH_SUFFIX = 'npy'

    def __init__(self, dataset_dir, stereo=False, rgbd=False, cache=False, pxl_prompt_h=None, pxl_prompt_w=None, pxl_prompt_seed=0, force_img_shape=None, force_video_len=None):
        '''
        :param dataset_dir: the root of dataset (where all videos are stored)
        :param rgb_only: whether only load rgb images or not (for rgb-d or stereo dataset)
        :param cache: whether to cache data once loaded, can turn on this option when testing inference speed
        '''
        self._cfg = dict(
            stereo=stereo,
            rgbd=rgbd,
            pxl_prompt_h=pxl_prompt_h,
            pxl_prompt_w=pxl_prompt_w,
            force_img_shape=force_img_shape,
            force_video_len=force_video_len,
        )

        self._dataset_dir = pathlib_file(dataset_dir)
        self._stereo = stereo
        self._num_cam = 2 if self.stereo else 1
        self._rgbd = rgbd
        self._cache = cache
        self._pxl_prompt = None
        if (pxl_prompt_h is not None) and (pxl_prompt_w is not None):
            self._pxl_prompt = (pxl_prompt_h, pxl_prompt_w)
        self._pxl_prompt_seed = pxl_prompt_seed
        self._force_img_shape = force_img_shape
        self._force_video_len = force_video_len
        self._scale = None

        self._video_dir = None
        self._rgb_fs = None
        self._gt_depth_fs = []

        self._img_shape = None

        # object canonical model
        self._can_mesh = None
        self._can_pcd = None

        if cache:
            self._imgs = dict()
            self._cam_params = dict()
            self._gt_segs = dict()
            self._gt_poses = dict()
            self._gt_depths = dict()

    def _init_gt_depth_fs(self):
        self._gt_depth_fs = get_all_files_with_suffix(self._video_dir.joinpath(self.GT_DEPTH_DIR), self.GT_DEPTH_SUFFIX, include_patterns=self.GT_DEPTH_PREFIX)
        if len(self._gt_depth_fs) != len(self.rgb_fs[0]):
            logger.warning(f'Cannot load gt_depth files from {self._video_dir.joinpath(self.GT_DEPTH_DIR).resolve().as_posix()}, because #files ({len(self._gt_depth_fs)}) doesn\'t match with rgb ({len(self)}). SO NOT LOADING ANY GT DEPTH FILES!')
            self._gt_depth_fs = []

    @property
    def cfg(self):
        self._cfg['pxl_prompt_h'], self._cfg['pxl_prompt_w'] = self.pxl_prompt
        return self._cfg

    @property
    def rgbd(self):
        return self._rgbd

    @property
    def stereo(self):
        return self._stereo

    @property
    def img_type(self):
        if self.stereo:
            return StringName.stereo_rgbd if self.rgbd else StringName.stereo
        else:
            return StringName.rgbd if self.rgbd else StringName.rgb

    @property
    def rgb_fs(self):
        return self._rgb_fs

    @property
    def video_dir(self):
        return self._video_dir

    @property
    def rel_video_dir(self):
        return self.video_dir.relative_to(self._dataset_dir)

    @property
    def scale(self):
        if self._scale is None:
            self._load_img(0)
        return self._scale

    def rescale(self, x):
        if self.scale[0] != 1 or self.scale[1] != 1:
            x = cv2.resize(x, (self._force_img_shape[1], self._force_img_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        return x

    def __len__(self):
        '''
        :return: number of frames in the video
        '''
        return len(self.rgb_fs[0]) if (self._force_video_len is None) else self._force_video_len

    def get_frame(self, idx, img=True, cam_params=True, gt_seg=False, gt_pose=False, gt_depth=False, gt_mesh=False, gt_pcd=False):
        '''
        :param idx: index of frame
        :param gt_seg: ground truth object mask
        :param gt_pose: ground truth object pose
        :param gt_depth: ground truth object depth
        :param gt_mesh: object mesh at the idx-th frame
        :param gt_pcd: object point cloud at the idx-th frame
        :return: a dictionary containing:
            img: a list (of length 2 if self.stereo else 1) of dictionaries of {'rgb': np array (H, W, 3)} (if self.rgbd is False) or {'rgb': np array (H, W, 3), 'd': np array (H, W)} (if self.rgbd is True), if img is True, else None
            cam_params: dictionaries of {'intr': np array (3 ,3)} (if self.stereo is False) or {'intr': np array (3 ,3), 'extr': np array (4, 4) (CAM_0^{-1} @ CAM_1)} (if self.stereo is True), if cam_params is True, else None
            gt_seg: np boolean arrays of shape (H, W) if gt_seg is true, else None
            gt_pose: np array of SE(3) matrix (shape (4, 4)) if gt_pose is true, else None
            gt_depth: np array with shape (H, W), gt_depth[h, w] = pixel_depth if (h, w) belongs to the object else -1, if gt_depth is true, else None
            gt_mesh: object mesh at the idx-th frame if gt_mesh is true, else None
            gt_pcd: object point cloud at the idx-th frame if gt_pcd is true, else None
        '''
        ret = dict(
            img=self._load_img(idx) if img else None,
            cam_params=self._load_cam_params(idx) if cam_params else None,
            gt_seg=self._load_gt_seg(idx) if gt_seg else None,
            gt_pose=self._load_gt_pose(idx) if gt_pose else None,
            gt_depth=self._load_gt_depth(idx) if gt_depth else None,
            gt_mesh=self._load_gt_mesh(idx) if gt_mesh else None,
            gt_pcd=self._load_gt_pcd(idx) if gt_pcd else None,
        )
        return ret

    @property
    def img_shape(self):
        if self._img_shape is not None:
            return self._img_shape
        self._load_img(0)
        return self._img_shape

    @property
    def pxl_prompt(self):
        '''
        :return: tuple (h, w), the pixel prompt at 0-th frame
        '''
        if self._pxl_prompt is not None:
            return self._pxl_prompt
        seg_0 = self._load_gt_seg(0)
        with FixedSeed(self._pxl_prompt_seed):
            pxl_prompt = np.random.choice(seg_0.shape[0] * seg_0.shape[1], p=(seg_0 / seg_0.sum()).reshape(-1))
        self._pxl_prompt = (pxl_prompt // seg_0.shape[1], pxl_prompt % seg_0.shape[1])
        assert seg_0[self._pxl_prompt[0], self._pxl_prompt[1]] > 0
        return self._pxl_prompt

    @property
    @abstractmethod
    def can_mesh_path(self):
        return ''

    def load_can_mesh(self):
        if self._can_mesh is not None:
            return self._can_mesh

        self._can_mesh = trimesh.load(self.can_mesh_path)
        return self._can_mesh

    def load_can_pcd(self):
        if self._can_pcd is None:
            self._can_pcd = np.array(self.load_can_mesh().vertices)
        return self._can_pcd

    @abstractmethod
    def _read_img(self, idx):
        '''
        :param idx: frame index
        :return: read image from disk
        '''
        return

    def _load_img(self, idx):
        '''
        :param idx: frame index
        :return: load the idx-th image
        '''
        if self._cache:
            if idx in self._imgs:
                return self._imgs[idx]

        ret = self._read_img(idx)
        if self._scale is None:
            if self._force_img_shape is None:
                self._scale = (1., 1.)
            else:
                self._scale = (float(self._force_img_shape[0]) / float(ret[0]['rgb'].shape[0]), float(self._force_img_shape[1]) / float(ret[0]['rgb'].shape[1]))
        for i in range(len(ret)):
            ret[i]['rgb'] = self.rescale(ret[i]['rgb'])
            if self.rgbd:
                ret[i]['d'] = self.rescale(ret[i]['d'])

        if self._cache:
            self._imgs[idx] = deepcopy(ret)

        if self._img_shape is None:
            self._img_shape = ret[0]['rgb'].shape[:2]
        return ret

    @abstractmethod
    def _read_cam_params(self, idx):
        return

    def _load_cam_params(self, idx):
        if self._cache:
            if idx in self._cam_params:
                return self._cam_params[idx]

        ret = self._read_cam_params(idx)
        ret['intr'] = scale_cam_mat(ret['intr'], sc_x=self.scale[1], sc_y=self.scale[0])

        if self._cache:
            self._cam_params[idx] = deepcopy(ret)
        return ret

    @abstractmethod
    def _read_gt_seg(self, idx):
        '''
        :param idx: frame index
        :return: read segmentation from disk
        '''
        return

    def _load_gt_seg(self, idx):
        '''
        :param idx: frame index
        :return: load the idx-th segmentation
        '''
        if self._cache:
            if idx in self._gt_segs:
                return self._gt_segs[idx]

        ret = self.rescale(self._read_gt_seg(idx).astype(np.uint8)).astype(np.bool_)

        if self._cache:
            self._gt_segs[idx] = np.copy(ret)
        return ret

    @abstractmethod
    def _read_gt_pose(self, idx):
        return

    def _load_gt_pose(self, idx):
        if self._cache:
            if idx in self._gt_poses:
                return self._gt_poses[idx]

        ret = self._read_gt_pose(idx)

        if self._cache:
            self._gt_poses[idx] = np.copy(ret)
        return ret

    def _read_gt_depth(self, idx):
        if idx < len(self._gt_depth_fs):
            ret = np.load(self._gt_depth_fs[idx].as_posix())
        else:
            ret = self._render_gt_depth(idx)
        return ret

    def _load_gt_depth(self, idx):
        if self._cache:
            if idx in self._gt_depths:
                return self._gt_depths[idx]

        ret = self.rescale(self._read_gt_depth(idx))

        if self._cache:
            self._gt_depths[idx] = np.copy(ret)
        return ret

    def _render_gt_depth(self, idx):
        data = self.get_frame(idx, img=False, cam_params=True, gt_pose=True)
        cam_params, gt_pose = data['cam_params']['intr'], data['gt_pose']
        _, d = render_meshes_to_img(self.img_shape[0], self.img_shape[1],
                                    [(self.load_can_mesh(), gt_pose)],
                                    cam_params[0][0], cam_params[1][1], cam_params[0][2], cam_params[1][2],
                                    )
        return d

    def _load_gt_mesh(self, idx):
        gt_pose = self.get_frame(idx, img=False, cam_params=False, gt_pose=True)['gt_pose']
        if gt_pose is None:
            return None
        mesh = deepcopy(self.load_can_mesh())
        mesh.apply_transform(gt_pose)
        return mesh

    def _load_gt_pcd(self, idx):
        gt_pose = self.get_frame(idx, img=False, cam_params=False, gt_pose=True)['gt_pose']
        if gt_pose is None:
            return None
        pcd = np.copy(self.load_can_pcd())
        pcd = apply_SE3(gt_pose, pcd)
        return pcd
