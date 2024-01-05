import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh

from core.utils.network_util import MotionBasisComputer
from core.nets.occnerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp

from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL
import os
import numpy as np
from copy import deepcopy
from torchvision import models

from .knn import knn as fast_knn 

from torch_cluster import fps as cluster_fps
from pytorch3d.ops.points_normals import estimate_pointcloud_normals


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


class Network(nn.Module):
    def __init__(self, avg_betas=None):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips)
        self.non_rigid_mlp = \
            nn.DataParallel(
                self.non_rigid_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, self.cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn


        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)


    def generate_neural_points(self, avg_betas):
        if avg_betas is not None:
            self.smpl = SMPL(sex='neutral', model_dir='./third_parties/smpl/models')
            verts, joints  = self.smpl(np.zeros(72,), avg_betas if avg_betas is not None else np.zeros(10,))
            base_mesh = trimesh.Trimesh(vertices=verts,#smpl_output.vertices[0].detach(),
                                        faces=self.smpl.faces,
                                        process=False,
                                        maintain_order=True)
            vertex_normals = base_mesh.vertex_normals

            
            min_xyz = np.min(joints, axis=0) - cfg.bbox_offset
            max_xyz = np.max(joints, axis=0) + cfg.bbox_offset
            self.bound = np.max(np.abs(list(min_xyz) + list(max_xyz)))
            self.detailed_bound = torch.tensor([list(min_xyz), list(max_xyz)])
        else:
            self.bound = None

        self.point_base = nn.Parameter(torch.tensor(verts).float(), requires_grad=False)
        self.point_dist = nn.Parameter(torch.zeros(verts.shape[0], 1).float(), requires_grad=True)
        self.point_dist.data.uniform_(-1e-4, 1e-4)

        # fps
        down_ratio = 1.
        self.fps_index = []
        for _ in range(3):
            down_ratio = down_ratio / 4
            index = cluster_fps(torch.tensor(verts), ratio=down_ratio)
            self.fps_index.append(index)


        self.point_counter =  nn.Parameter(torch.ones(verts.shape[0]), requires_grad=False)
        self.point_norms = torch.tensor(vertex_normals)


        ranges = torch.cumsum(torch.tensor([0, self.point_base.shape[0], self.fps_index[0].shape[0], self.fps_index[1].shape[0], self.fps_index[2].shape[0]]), 0)
        self.ranges_y = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous()
        self.slices_x = (torch.arange(0, 4) + 1).int().view(4,)
        self.slices_y = (torch.arange(0, 4) + 1).int().view(4,)
        self.offset = ranges[:-1].view(4,1)
        

        # canonical mlp 
        skips = []
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=self.cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips,
                bound=self.bound,
                detailed_bound=self.detailed_bound)
        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.primary_gpus[0])


    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self

    @property
    def point_cloud(self):
        return self.point_base + self.point_dist


    def get_point_normal(self):
        return estimate_pointcloud_normals(self.point_cloud.unsqueeze(0), neighborhood_size = 10)[0]

    def _query_mlp(
            self,
            pos_xyz,
            rays_d,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        rays_d_flat = torch.reshape(rays_d, [-1, rays_d.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        rays_d_flat=rays_d_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            rays_d_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        raws = []

        
        point_counter = self.point_counter.detach().to(pos_flat.device)# / torch.max(self.point_counter)[0]
        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]
            d = rays_d_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            # knn on multiscale point clouds
            k = 10
            with torch.no_grad():
                ranges = torch.cumsum(torch.tensor([0, xyz.shape[0], xyz.shape[0], xyz.shape[0], xyz.shape[0]]), 0)
                ranges_x = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous()
                knn_idxs = fast_knn(xyz.repeat(4,1),
                                torch.cat([self.point_base, self.point_base[self.fps_index[0]], self.point_base[self.fps_index[1]], self.point_base[self.fps_index[2]]], dim=0),
                                k,
                                #distance_limit = 0.1,
                                return_distance = False,
                                padding_mode= "empty",
                                ranges_x=ranges_x.to(xyz.device), 
                                slices_x=self.slices_x.to(xyz.device), 
                                ranges_y=self.ranges_y.to(xyz.device), 
                                slices_y=self.slices_y.to(xyz.device)
                                )

                offset = self.offset.expand(-1, xyz.shape[0]).contiguous().view(-1,1) # N, 1
                knn_idxs -= offset.to(knn_idxs)
        
                knn_idxs = torch.stack([knn_idxs[:xyz.shape[0]], self.fps_index[0].to(xyz.device)[knn_idxs[xyz.shape[0]:2*xyz.shape[0]]], 
                            self.fps_index[1].to(xyz.device)[knn_idxs[2*xyz.shape[0]:3*xyz.shape[0]]], self.fps_index[2].to(xyz.device)[knn_idxs[3*xyz.shape[0]:]]], dim=1).to(xyz.device) # N, 4, k


            N = xyz.shape[0]
            
            knn_att = point_counter[knn_idxs].view(N,-1,1)
            point_norms = self.point_norms.to(xyz.device)[knn_idxs[:,0]].view(N, k, -1) # N, K, 3

            point_cloud = self.point_cloud.to(xyz)
            point_base = self.point_base.to(xyz)
            kidx = fast_knn(point_cloud.float(),
                    point_base,
                    3,
                    return_distance = False,
                    padding_mode= "empty")

            # determine sdf
            knn_base = point_base[kidx].view(-1, 3, 3)
            direction_from_surface = point_cloud.unsqueeze(1) - knn_base  # N, S, 3

            norms = self.point_norms.to(xyz.device)[kidx].view(point_cloud.shape[0], 3, -1) # N, 3

            att = torch.abs(F.cosine_similarity(direction_from_surface, norms, dim=-1)).unsqueeze(-1) # N, k, 1
            knn_base = torch.sum(att * knn_base, dim=1) / torch.sum(att, dim=1) # N, 3

            inside = torch.einsum('ijk,ijk->ij', direction_from_surface.float(), norms.float()) < 0
            inside = torch.sum(inside, dim=1) > 3 * 0.5

            dist = torch.mean(torch.norm(direction_from_surface, dim=-1), dim=1, keepdim=True) # N, 1
            dist[inside] *= -1 # inside is < 0
            

            xyz_embedded = pos_embed_fn(xyz)

            raws += [self.cnl_mlp(
                        xyz=xyz,
                        xyz_embedded=xyz_embedded,
                        knn_points=self.point_base[knn_idxs[:,0]].view(N, k, -1),
                        point_norms=point_norms,
                        knn_att=knn_att,
                        point_cloud=knn_base,
                        point_sdf=dist, 
                        knn_idxs=knn_idxs,
                        learnable_points=point_cloud,
                        )]

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.softplus):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0] 

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        term_point = torch.argmax(alpha, dim=1, keepdim=True)

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map, term_point # N, S


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        if ray_batch.shape[-1] > 8:
            ray_feats = ray_batch[:,8:]
            return rays_o, rays_d, near, far, ray_feats
        return rays_o, rays_d, near, far, None


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far, ray_feats = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        N, S = pts.shape[0], pts.shape[1]
        batched_ray_d = rays_d.unsqueeze(-2).expand(-1,pts.shape[-2],-1) # N, S, 3

        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                rays_d=batched_ray_d,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        
        rgb_map, acc_map, _, depth_map, term_point = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        dist = raw[..., 4:] # N, S, 1
        alpha = raw[..., [3]]
        feats = raw[..., :3]
        
        if self.training:
            k = 10

            depth_mask = depth_map.detach() > 0.5 # 

            dist_in = (dist < 0.).float()
            dist_out = (dist > 0.3).bool()

            alpha[dist_out] *= 0.


            comp_loss = dist_in.detach() * torch.exp(torch.clamp(-F.relu(alpha), min=-10, max=0))

            comp_loss = comp_loss.squeeze(-1) * 10.

            # visibility attention
            if torch.sum(depth_mask) > 1:
                term_point = term_point[depth_mask].detach()
                term_pts = cnl_pts.view(N, S, -1)[depth_mask].detach() # n, S

                term_pts = batched_index_select(term_pts, 1, term_point).squeeze(1) # N

                knn_index = fast_knn(term_pts.detach().float(),
                                    self.point_cloud.float(),
                                    k,
                                    return_distance = False,
                                    padding_mode= "empty")

                N, k = knn_index.shape[0], knn_index.shape[1] 

                knn_index = knn_index.view(-1) # 
                self.point_counter[knn_index.to(self.point_counter.device)] += 1. # accumulate visibility
        else:
            comp_loss = torch.zeros(1,1)


        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map,
                'comp_loss': comp_loss}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        
        if self.training:
            packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)
        else:
            packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)
        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            if k == 'comp_loss':
                all_ret[k] = all_ret[k].view(-1)
            else:
                k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret