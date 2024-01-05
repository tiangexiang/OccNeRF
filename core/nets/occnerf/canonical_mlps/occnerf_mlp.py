import torch
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.occnerf.gridencoder import GridEncoder
from core.nets.occnerf.shencoder import SHEncoder
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

import torch.nn.functional as F
from copy import deepcopy
from pytorch3d.ops.points_normals import estimate_pointcloud_normals


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, skips=None, bound=1, geo_feat_dim=63,
                 **_):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        self.bound = bound

        self.encoder = GridEncoder(input_dim=4, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048*bound, gridtype='hash', align_corners=False)
        #self.dir_encoder = SHEncoder(input_dim=3, degree=4)
        self.neural_point_dim = 32 + 32#self.encoder.output_dim

        pts_block_mlps = [nn.Linear(1 + 3 + 32 + 32, self.mlp_width), nn.ReLU(inplace=True)]


        layers_to_cat_input = []
        for i in range(mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU(inplace=True)]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True)]

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        self.geo_linear = nn.Sequential(nn.Linear(mlp_width, 64 + 1))
        initseq(self.geo_linear)

        ################ color
        pts_block_mlps = [nn.Linear(64 +  32 + 32 + 3, mlp_width), nn.ReLU(inplace=True)]

        layers_to_cat_input = []
        for i in range(mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU(inplace=True)]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True)]

        self.rgb_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.rgb_linears)

        self.output_linear = nn.Sequential(nn.Linear(mlp_width, 3))
        initseq(self.output_linear)


    def simple_agg(self, atts, feats):
        # if type(knn_points) is not list:
        #     knn_points = [knn_points]
        #     feats = [feats]

        # # x: N, K, C
        # if xyz.dim() == 2:
        #     xyz = xyz.unsqueeze(1) 
        
        # feats = feats.view()#torch.cat(feats, dim=1) # N, k * level, c
        # atts = torch.cat(knn_att, dim=1)

        

        #mmin = atts.min(dim=1, keepdim=True)[0].detach().clone()
        #print(mmin.shape, atts.shape)
        #mmin2 = torch.empty_like(mmin).data
        #mmin2.data = mmin.data

        #test_max = atts.max(dim=1, keepdim=True)[0].detach().clone()
        #print(atts[100,10])

        #new_att = atts.detach() - atts.min(dim=1, keepdim=True)[0].detach()
        #atts += -1.
        atts += 1. - atts.min(dim=1, keepdim=True)[0]
        atts /= atts.max(dim=1, keepdim=True)[0]
        var = torch.var(atts, dim=1)
        atts = F.softmax(atts, dim=1)
        

        
        # knn_points = torch.stack(knn_points, dim=-1)


        # xyz = xyz.unsqueeze(-1)
        #diff = 1/(torch.norm(xyz - knn_points, dim=-2) + 1e-8) # N, k, level
        #denorm = torch.sum(diff, dim=-2, keepdim=True) + 1e-8 # N, 1, level

        agg = torch.sum(atts.detach() * feats, dim=1) # N, C, level
      
        return agg.view(agg.shape[0], -1), var


    def color_agg(self, knn_colors, feats):
        reference_color = knn_colors[0][:,[0],:] # N, 1, c

        feats = torch.stack(feats, dim=2)[:,1:,:,:].view(feats[0].shape[0], -1, feats[0].shape[-1]) # N, k * level, c

        atts = torch.stack(knn_colors, dim=2)[:,1:,:,:].view(feats.shape[0], -1, 3)

        atts = F.softmax(F.cosine_similarity(atts, reference_color, dim=-1), dim=1) # N, k, 1
        #print(atts.shape, feats.shape, reference_color.shape)
        agg = torch.sum(atts.detach().unsqueeze(-1) * feats, dim=1) # N, C, level
      
        return agg.view(agg.shape[0], -1)

    def forward(self, xyz, xyz_embedded, knn_points, point_norms, knn_att, point_cloud, point_sdf, knn_idxs, learnable_points, **_):

        N, k = knn_idxs.shape[0], knn_idxs.shape[2] 
        
        with torch.no_grad():
            direction_from_surface = xyz.unsqueeze(1) - knn_points  # N, S, 3

            # norms = estimate_pointcloud_normals(point_cloud.unsqueeze(0), neighborhood_size = 10)[0]
            # point_norms = norms[knn_idx[:,0,:3]].view(N, 3, -1) # N, 3

            inside = torch.einsum('ijk,ijk->ij', direction_from_surface.double(), point_norms.double()) < 0
            inside = torch.sum(inside, dim=1) > k * 0.5

            dist = torch.mean(torch.norm(direction_from_surface, dim=-1), dim=1, keepdim=True).detach() # N, 1
            dist[inside] *= -1 # inside is < 0
            normed_dist = torch.clamp((dist + 0.2) / 0.5, 0.0, 1.0)


        
        #print(var.shape)
      
        #knn_points = knn_points[:,:3] 
        knn_points = (knn_points + self.bound) / (2 * self.bound) 
        att = torch.abs(F.cosine_similarity(direction_from_surface[:,:3], point_norms[:,:3], dim=-1)).unsqueeze(-1) # N, k, 1
        knn_points = torch.sum(att * knn_points[:,:3], dim=1) / torch.sum(att, dim=1) # N, 3
        h = self.encoder(torch.cat((knn_points, normed_dist), dim=-1).float(), bound=None)



        point_cloud = (point_cloud + self.bound) / (2 * self.bound) 
        point_sdf = torch.clamp((point_sdf + 0.2) / 0.8, 0.0, 1.0)
        
        knn_feats = self.encoder(torch.cat((point_cloud, point_sdf), dim=-1).float(), bound=None)
        knn_feats = torch.cat((knn_feats, learnable_points.float()), dim=-1) # N, 32+3
        knn_feats = knn_feats[knn_idxs].view(N, -1, knn_feats.shape[-1])

        knn_feats, var = self.simple_agg(knn_att, knn_feats) # N, c

        #h = xyz_embedded
        encoded_h = h

        h = torch.cat([knn_feats, var, h], dim=-1).float()

        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

        h = self.geo_linear(h)
        sigma = h[..., [0]]

        h = torch.cat([h[...,1:], knn_feats, encoded_h], dim=-1)

        for i, _ in enumerate(self.rgb_linears):
            h = self.rgb_linears[i](h)

        h = self.output_linear(h)


        return torch.cat((h, sigma, dist.detach()), dim=-1)

    # def forward(self, xyz, xyz_embedded, point_cloud, point_feats, knn_idx, **_):

    #     N, k = knn_idx.shape[0], knn_idx.shape[1] 
    #     knn_feats = point_feats[knn_idx].view(N, k, -1)
    #     knn_points = point_cloud[knn_idx].view(N, k, -1)
    #     knn_feats = self.simple_agg(xyz, knn_points, knn_feats)

    #     dist = torch.min(torch.norm(knn_points - xyz.unsqueeze(1), dim=-1), dim=1, keepdim=True)[0].detach() # N, 1

    #     #h = h + point_feats
    #     h = torch.cat([dist, xyz_embedded], dim=-1)
    #     sigma = self.geo_linear(h)
    #     sigma = trunc_exp(sigma)
        
    #     #print('???')

    #     # xyz N, 3
    #     h = self.encoder(xyz, bound=self.bound)
    #     # h = h[knn_idx].view(N, k, -1)
    #     # h = self.simple_agg(xyz, knn_points, h)

    #    #h = h + knn_feats

    #     #h = point_feats
    #     encoded_h = h

    #     h = torch.cat([h, xyz_embedded, knn_feats], dim=-1)

    #     for i, _ in enumerate(self.pts_linears):
    #         # if i in self.layers_to_cat_input:
    #         #     h = torch.cat([pos_embed, h], dim=-1)
    #         h = self.pts_linears[i](h)

    #     # d = self.dir_encoder(dir)
    #     h = torch.cat([encoded_h, knn_feats, xyz_embedded], dim=-1)

    #     for i, _ in enumerate(self.rgb_linears):
    #         # if i in self.layers_to_cat_input:
    #         #     h = torch.cat([pos_embed, h], dim=-1)
    #         h = self.rgb_linears[i](h)

    #     h = self.output_linear(h)

    #     # def occ_eval_fn(x):
    #     #     return 

    #     # self.occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

    #     return torch.cat((h, sigma, encoded_h.detach()), dim=-1)


