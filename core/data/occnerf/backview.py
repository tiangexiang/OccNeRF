import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.camera_util import \
    rotate_camera_by_frame_idx, \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox
from core.utils.file_util import list_files, split_path

from configs import cfg


class Dataset(torch.utils.data.Dataset):
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }

    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            skip=1,
            bgcolor=None,
            src_type="zju_mocap",
            **_):

        print('[Dataset Path]', dataset_path) 

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints, 
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')

        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        framelist = self.load_train_frames() 
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]  

        # self.train_frame_idx = cfg.freeview.frame_idx
        #print(f' -- Frame Idx: {self.train_frame_idx}')

        self.total_frames = len(self.framelist)
        self.period = cfg.render_frames
        self.reference_angle_idx = cfg.render_frames // 2
        print(f' -- Total Rendered Frames: {self.total_frames}')

        # self.train_frame_name = framelist[self.train_frame_idx]
        # self.train_camera = cameras[framelist[self.train_frame_idx]]
        # self.train_mesh_info = mesh_infos[framelist[self.train_frame_idx]]

        self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]
        self.keyfilter = keyfilter
        self.src_type = src_type

        self.reference_extrinsics = None

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'betas': self.mesh_infos[frame_name]['betas'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            # 'dst_tpose_joints': \
            #     self.mesh_infos[frame_name]['dapose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    def get_backview_camera(self, frame_idx, frame_name, trans=None):
        E = rotate_camera_by_frame_idx(
                extrinsics=self.cameras[frame_name]['extrinsics'], 
                frame_idx=frame_idx,
                period=self.period,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        K = self.cameras[frame_name]['intrinsics'].copy()
        K[:2] *= cfg.resize_img_scale
        return K, E

    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        if 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        #img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        img = orig_img
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                             fx=cfg.resize_img_scale,
                             fy=cfg.resize_img_scale,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # frame_name = self.train_frame_name
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name
        }
       # print('rendering frame name:', frame_name)

        bgcolor = np.array(self.bgcolor, dtype='float32')

        img, _ = self.load_image(frame_name, bgcolor)
        img = img / 255.

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)

        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Rh = dst_skel_info['Rh']
        dst_Th = dst_skel_info['Th']

        if self.reference_extrinsics is None:
            K, E = self.get_backview_camera(
                            frame_idx=self.reference_angle_idx,
                            frame_name=frame_name,
                            #trans=dst_Th)
                            trans=None)
            self.reference_extrinsics = K, E
        else:
            K, E = self.reference_extrinsics
        
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_Rh,
                Th=dst_Th)
        R = E[:3, :3]
        T = E[:3, 3]

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })                                    

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = \
                self.motion_weights_priors.copy()

        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })


        return results
