import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
import os
#from smplx import SMPL

#try:
from third_parties.smpl.smpl_numpy import SMPL
# except:
#     import sys
#     sys.path.insert(0, '/vision/u/xtiange/human/nerf/humannerf')
#     from third_parties.smpl.smpl_numpy import SMPL

import itertools
import numpy as np

from torch_cluster import knn as cluster_knn
from torch_cluster import fps as cluster_fps

from easy_kpconv.ops.knn import knn as kpknn 

from copy import deepcopy


# input: 1, N, 3
# dim: 0 < scalar
# index: B x M
def batched_index_select(input, dim, index):
    B, N = index.shape[0], index.shape[1]
    views = [index.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape) # 1, N, 3
    expanse[0] = -1
    expanse[dim] = -1 # -1, -1, 3 # views: B, -1, 1
    print(views)
    index = index.view(views).expand(expanse) # -1, -1, 3
    print(index.shape, input.shape)
    res = torch.gather(input, dim, index).view(B, N, -1)
    print(B, N, res.shape)
    return res


class NeuralPoint(nn.Module):
    def __init__(self, verts, pos_embed_fn=None, num_levels=1, learn_points=True, 
                 learn_confidence=False, down_ratio=0.25, dim=16,
                 enable_fast_knn=False, sigma_dim=0, normals=None, generate_feature=False):
        super(NeuralPoint, self).__init__()
        self.enable_fast_knn = enable_fast_knn

        self.learn_confidence = learn_confidence

        verts = torch.tensor(verts)

        self.point_cloud = nn.ParameterList()
        self.point_cloud.append(nn.Parameter(verts.float(), requires_grad=learn_points))

        # v18 learn confidnence instead of position
        # self.point_cloud = nn.ParameterList()
        # self.point_cloud.append(nn.Parameter(torch.empty_like(verts).float(), requires_grad=False))
        
        self.fps_index = []

        if normals is not None:
            self.normals = [torch.tensor(normals)]
        
        low_res_verts = verts
        #self.point_cloud[-1].data = low_res_verts.clone()
        for i in range(1, num_levels):
            index = cluster_fps(low_res_verts, ratio=down_ratio)

            if normals is not None:
                self.normals.append(self.normals[-1][index])

            self.fps_index.append(index)
            low_res_verts = low_res_verts[index]
            self.point_cloud.append(nn.Parameter(low_res_verts.float(), requires_grad=learn_points))
            #self.point_cloud[-1].data = low_res_verts.clone()
            print('low res point shape', self.point_cloud[-1].shape)
        print(len(self.point_cloud))

        self.neural_point = nn.ParameterList()
        for i in range(len(self.point_cloud)):
            if pos_embed_fn is None:
                self.neural_point.append(nn.Parameter(torch.empty(self.point_cloud[i].data.shape[0], dim + sigma_dim), requires_grad=True))
                self.neural_point[-1].data.uniform_(-1e-4, 1e-4)
            else:
                self.neural_point.append(nn.Parameter(pos_embed_fn(self.point_cloud[i].data), requires_grad=True))

        if learn_confidence:
            self.point_conf = nn.ParameterList()
            for i in range(len(self.point_cloud)):
                self.point_conf.append(nn.Parameter(torch.empty(self.point_cloud[i].data.shape[0], 1), requires_grad=True))
                self.point_conf[-1].data.uniform_(-1e-4, 1e-4)

        # register
        # self.register_parameter('point_cloud', verts_list)
        # self.register_parameter('neural_point', neural_point_list)

        self.output_dim = sum([self.neural_point[i].shape[-1] for i in range(num_levels)])
        self.res = 100
        self.mmin = torch.min(self.point_cloud[0], dim=0)[0] - 0.05
        self.mmax = torch.max(self.point_cloud[0], dim=0)[0] + 0.05

        

        if enable_fast_knn:
            
            print('Start processing fast knn...')
            self.index_book, self.flatten_grid = self.fast_knn_preprocess(self.res, 30)
            print('processing fast knn finished!')


    def fast_knn_preprocess(self, res=100, num=30):
        mmin = self.mmin
        mmax = self.mmax
        x_grid = torch.arange(mmin[0].item(), mmax[0].item(), (mmax[0].item() - mmin[0].item()) / res).view(res, 1, 1).expand(-1, res, res)
        y_grid = torch.arange(mmin[1].item(), mmax[1].item(), (mmax[1].item() - mmin[1].item()) / res).view(1, res, 1).expand(res, -1, res)
        z_grid = torch.arange(mmin[2].item(), mmax[2].item(), (mmax[2].item() - mmin[2].item()) / res).view(1, 1, res).expand(res, res, -1)
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1) # res, res, res, 3
        flatten_grid = grid.view(-1, 3)
        print(flatten_grid[0], flatten_grid[-1])

        index_book = []
        for i in range(len(self.point_cloud)):
            knn_idx = self.knn(flatten_grid, self.point_cloud[i], k=1).transpose(0,1).contiguous().detach().cuda()
            N, k = knn_idx.shape[0], knn_idx.shape[1]
            index_book.append(knn_idx)
            # debug
            debug_idx = 11 * self.res**2 + 51 * self.res + 10
            debug = self.point_cloud[i].cuda()[knn_idx.view(-1)].view(N, k, -1)
            diff = torch.sum(torch.abs(flatten_grid.cuda()[debug_idx].unsqueeze(0) - debug[debug_idx]))
            print('debug diff', diff)

        return index_book, flatten_grid

    # torch cluster
    # @staticmethod
    # def knn(query, data, k):
    #     index = cluster_knn(data.double(), query.double(), k) # 2.3
    #     index = index[1].view(-1,k)
    #     return index

    @staticmethod
    def knn(query, data, k):
        return kpknn(query.float(),
                    data.float(),
                    k,
                    #distance_limit = 0.1,
                    return_distance = False,
                    padding_mode= "empty")
        

    def fast_knn(self, query, data_index, k):
        interval = (self.mmax - self.mmin) / self.res # 3, 
        interval = interval.view(1, 3).to(query)
        print(query.shape, self.mmin.shape, interval.shape)
        print(self.mmin)
        pos = (query - self.mmin.to(query).unsqueeze(0)) / interval
        pos = torch.floor(pos).long() # N, 3
        print(pos)

        # dummy_point = torch.zeros_like(self.point_cloud[data_index][[0],:]) - 1e9
        # data = torch.cat((dummy_point, self.point_cloud[data_index]), dim=0)

        # sample grid
        # corase results
        flatten_pos1 = pos[:,0] * self.res**2 + pos[:,1] * self.res + pos[:,2] # N,
        print(flatten_pos1, 'check')
        flatten_pos = torch.argmin(torch.mean(torch.abs(query.cuda() - self.flatten_grid.cuda() ), dim=-1), dim=0).view(1)
        print(flatten_pos, 'test')

        candidate_index = self.index_book[data_index][flatten_pos1] # N, num
        N, k = candidate_index.shape[0], candidate_index.shape[1]
        #batch_data = torch.index_select(self.point_cloud[data_index], 0, candidate_index.view(-1)).view(*candidate_index.shape, 3)
        batch_data = self.point_cloud[data_index][candidate_index.view(-1)].view(N, k, -1)
        #batch_data = batch_data[flatten_pos1] # N, num, 3

    
        batch_index = torch.arange(0, batch_data.shape[0]).long().to(query).view(-1,1).expand(-1,batch_data.shape[1]).contiguous().view(-1) # N, num
        batch_data = batch_data.view(-1, 3)

        # candidate_index = []
        # for i in [0, 1]:
        #     for j in [0, 1]:
        #         for k in [0, 1]:
        #             c_pos = pos
        #             c_pos[:,0] += i
        #             c_pos[:,1] += j
        #             c_pos[:,2] += k
        #             flatten_pos = c_pos[:,0] * self.res**2 + c_pos[:,1] * self.res + c_pos[:,2]
        #             candidate_index.append(self.index_book[data_index][flatten_pos]) # N, num

        # candidate_index = torch.cat(candidate_index, dim=-1) # N, num

        index = cluster_knn(batch_data.double(), query.double(), k, batch_x=batch_index, batch_y=torch.arange(0, query.shape[0]).long().to(query.device)) # 2.3
        index = index[1].view(-1,k).transpose(1,0)
        return index

    def query(self, xyz, k, return_index=False, point_cloud=None, return_normals=False, single_query=False):

        if point_cloud is None:
            point_cloud = self.point_cloud

        knn_featrues = []
        knn_points = []
        knn_confs = []
        knn_index = []
        knn_normals = []
        for i in range(len(point_cloud)):

            if not single_query or (single_query and i == 0):
                if self.enable_fast_knn:
                    knn_idx = self.fast_knn(xyz, i, k=k).detach()
                else:
                    knn_idx = self.knn(xyz, point_cloud[i], k=k).detach().to(xyz.device)
            knn_idx = knn_idx.view(-1, k)
            N, k = knn_idx.shape[0], knn_idx.shape[1]

            if N == 0:
                print('ERROR: no knn found!', knn_idx.shape, xyz.shape, total_elem, point_cloud.shape)
                return None

            if return_index:
                knn_index.append(knn_idx)

            if return_normals:
                knn_normals.append(self.normals[i].to(xyz.device)[knn_idx])
      
            knn_idx = knn_idx.view(-1) # N, k
            knn_featrues.append(self.neural_point[i][knn_idx].view(N, k, -1)) # N*k, C
            knn_points.append(point_cloud[i][knn_idx].view(N, k, -1))
            if self.learn_confidence:
                knn_confs.append(self.point_conf[i][knn_idx].view(N, k, -1))
        #knn_featrues = torch.cat(knn_featrues, dim=-1) # N*k, C*num_level

        if return_index:
            return knn_points, knn_featrues, knn_index

        if return_normals:
            return knn_points, knn_featrues, knn_normals

        if self.learn_confidence:
            return knn_points, knn_featrues, knn_confs
        else:
            return knn_points, knn_featrues


if __name__ == '__main__':
    smpl = SMPL(sex='neutral', model_dir=os.path.join('/vision/u/xtiange/human/nerf/humannerf', 'third_parties/smpl/models'))
    verts, joints  = smpl(np.zeros(72,), np.zeros(10,))
    # verts, faces = trimesh.remesh.subdivide(verts, smpl.faces)
    neuralpoint = NeuralPoint(verts, num_levels=1, down_ratio=0.5, dim=4, learn_points=False, enable_fast_knn=True).cuda()
    inp = (neuralpoint.mmin + 0.2).view(1,3).expand(-1, -1).cuda()
    knn_points, knn_featrues = neuralpoint.query(inp, 1)
    diff = torch.sum(torch.abs(inp.unsqueeze(1) - knn_points[0]))
    print(diff)
    