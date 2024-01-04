import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data
import trimesh
from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox
import warnings
from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL

from matplotlib import pyplot as plt

import torchvision.transforms as T

# try:
#     from .occlude import load_occluders, occlude_with_objects
# except:
#     pass


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            occlude=False,
            **_):

        print('[Dataset Path]', dataset_path) 

        self.occlude = hasattr(cfg, 'occlude') and cfg.occlude is True

        
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox, self.avg_betas = \
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
        
        print(f' -- Total Frames: {self.get_total_frames()}')
        

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode

        self.smpl = SMPL(sex='neutral', model_dir='./third_parties/smpl/models')

        self.img_processor = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])

        self.counter = 0

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        avg_betas = cl_joint_data['avg_betas'].astype('float32')

        return canonical_joints, canonical_bbox, avg_betas

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
        ret =  {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'betas': self.mesh_infos[frame_name]['betas'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            # 'dst_tpose_joints': \
            #     self.mesh_infos[frame_name]['dapose_joints'].astype('float32'),
            'joints': \
                self.mesh_infos[frame_name]['joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }
        if 'trans' in self.mesh_infos[frame_name]:
            ret.update({'trans': self.mesh_infos[frame_name]['trans'].astype('float32')})
        return ret

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far, select_inds
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
    # do dilation here to enlarge masking range
    def load_image(self, frame_name, bg_color, idx=None):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))


        if self.occlude and idx < cfg.occlusion.range:
            alpha_mask[:, cfg.occlusion.mid-cfg.occlusion.width//2:cfg.occlusion.mid+cfg.occlusion.width//2] *= 0

        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        o_img_shape = img.shape

        if hasattr(cfg, 'crop_image_scale') and cfg.crop_image_scale[0] != -1:
            mid_x, mid_y = img.shape[0] // 2, img.shape[1] //2
            dx, dy = cfg.crop_image_scale[0], cfg.crop_image_scale[1]
            img = img[mid_x-dx//2:mid_x+(dx-dx//2),mid_y-dy//2:mid_y+(dy-dy//2)]
            alpha_mask = alpha_mask[mid_x-dx//2:mid_x+(dx-dx//2),mid_y-dy//2:mid_y+(dy-dy//2)]

        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask, o_img_shape


    def get_total_frames(self):
        return len(self.framelist)

    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, near, far, patch_mask = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, near, far)
        
        targets = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, patch_mask, \
                target_patches, patch_masks, patch_div_indices

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):

        
        frame_name = self.framelist[idx]

        frame_idx = int(frame_name[-6:])
        results = {
            'frame_name': frame_name,
            'idx': frame_idx,
            'time': idx / len(self.framelist)
        }

        dst_skel_info = self.query_dst_skeleton(frame_name)

        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']

        
        dst_betas = dst_skel_info['betas']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        if 'trans' in dst_skel_info:
            warnings.warn('World to camera tanslation detected!')
            dst_trans = dst_skel_info['trans']
        else:
            dst_trans = None#np.zeros_like(dst_betas)[:3]

        # smpl
        verts, joints  = self.smpl(dst_poses, dst_betas, trans=dst_trans)
        

        if hasattr(cfg, 'upsample_pc') and cfg.upsample_pc:
            verts, joints = trimesh.remesh.subdivide(verts, self.smpl.faces)

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        # load img, mask, alpha blending the two
        img, alpha, o_img_shape = self.load_image(frame_name, bgcolor, idx=idx) 

        while np.sum(alpha) < 1: # at least 10 valid pixels in the patch
            return self.__getitem__(np.random.randint(0, self.get_total_frames()))

        img = (img / 255.).astype('float32')

        if hasattr(cfg, 'include_img') and cfg.include_img:
            results.update({
                'img': self.img_processor(img),
                'alpha': alpha
            })


        H, W = img.shape[0:2]

        # update pose and beta
        results.update({
            'poses': dst_poses,
            'betas': dst_betas,
            'Rh': cv2.Rodrigues(dst_skel_info['Rh'])[0].astype(np.float32),
            'Th': dst_skel_info['Th'],
            'joints': dst_skel_info['joints'],
            'verts': verts,
        })

        assert frame_name in self.cameras

        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        if hasattr(cfg, 'crop_image_scale') and cfg.crop_image_scale[0] != -1:
            mid_x, mid_y = o_img_shape[0] // 2, o_img_shape[1] //2
            dx, dy = cfg.crop_image_scale[0], cfg.crop_image_scale[1]
            
            K[0, -1]  = dx/2
            K[1, -1]  = dy/2


        K[:2] *= cfg.resize_img_scale

        E = self.cameras[frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)


        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        ray_alpha = alpha.reshape(-1, 3)[ray_mask]
        
        ######### update alpha mask for ocmotion evaluation #########
        results.update({
                'ray_alpha': ray_alpha
            })
        ######### update alpha mask for ocmotion evaluation #########

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, near, far, patch_mask, \
            target_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, 
                                       ray_img=ray_img, 
                                       near=near, 
                                       far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
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

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'patch_mask': patch_mask,
                    'target_patches': target_patches})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
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
