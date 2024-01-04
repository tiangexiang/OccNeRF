import sys
sys.path.insert(0, '/vision/u/xtiange/human/nerf/humannerf')
from third_parties.smpl.smpl_numpy import SMPL
import trimesh
import os
import numpy as np


import torch

import time

#from torchpq.index import IVFPQIndex # requires cupy
#from numba_kdtree import KDTree as numbaKDTree



#from pytorch3d.ops import ball_query


def knn(query, data, k):
    dist = torch.norm(data.unsqueeze(1) - query.unsqueeze(0), dim=2, p=None)
    knn = dist.topk(k, largest=False, dim=0)
    return knn.indices

print('start!')
smpl = SMPL(sex='neutral', model_dir=os.path.join('/vision/u/xtiange/human/nerf/humannerf', 'third_parties/smpl/models'))
verts, joints  = smpl(np.zeros(72,), np.zeros(10,))
verts, faces = trimesh.remesh.subdivide(verts, smpl.faces)
print(verts.shape)

#data = torch.randn(100000, 3).cuda() * 1000.
data = torch.tensor(verts).cuda()
# query = torch.zeros(100000, 3).cuda() * 1000.
# data = torch.zeros(100000, 3).cuda() * 1000.

query = torch.rand(1000000, 3).cuda()
query *= data.max() - data.min()
query -= data.min()


k = 20

names = ['baseline', 'cluster', 'scipy', 'kpconv', 'torchpq']
name = names[3]

if name == 'baseline':
    start = time.time()
    index = knn(query, data, k)

if name == 'cluster': # 0.96, 99.75
    from torch_cluster import knn as cluster_knn
    start = time.time()
    index = cluster_knn(data.double(), query.double(), k) # 2.3
    index = index[1].view(-1,20).transpose(1,0)

if name == 'scipy': # 0.03, 1.78
    from scipy.spatial import KDTree, cKDTree
    tree = cKDTree(data.cpu().numpy(), leafsize=20)
    start = time.time()
    _, index = tree.query(query.cpu().numpy(), k, distance_upper_bound=0.5)

if name == 'kpconv': # 0.015, 40.9
    from easy_kpconv.ops.knn import knn as kpknn 
    start = time.time()
    print('kpknn')
    index = kpknn(query.float(),
    data.float(),
    k,
    #distance_limit = 0.001,
    return_distance = False,
    padding_mode= "empty")
    #index = index.transpose(1,0)

if name == 'dgl':
    from dgl import knn_graph
    start = time.time()
    index = knn_graph(x, k, algorithm='bruteforce-sharemem')
    index = index[1].view(-1,20).transpose(1,0)
    



# query[0] += 101

# data[2000] += 100
# data[4000] += 100
# data[9211] += 100


#tree = cKDTree(data.cpu().numpy(), leafsize=20)
#tree = numbaKDTree(data.cpu().numpy(), leafsize=10)

# tree = IVFPQIndex(
#   d_vector=3,
#   n_subvectors=64,
#   n_cells=1024,
#   initial_size=2048,
#   distance="euclidean",
# )
# tree.train(data)


#knn = KNN(k=10, transpose_mode=True)




#_, index = tree.query(query.cpu().numpy(), 10, distance_upper_bound=0.5) # 1.00, 3.4
#topk_values, topk_ids = tree.topk(query, k=20)
# index = cluster_knn(data.double(), query.double(), k) # 2.3
# index = index[1].view(-1,20).transpose(1,0)

#print(query.shape, data.shape)

#index = cluster_knn(data, query, 20) # 0.94, 97



#_, indx = knn(data.unsqueeze(0), query.unsqueeze(0))  # 32 x 50 x 10
#print(index)
print(index.shape)
print('time:', time.time() - start)
print('memory:', torch.cuda.max_memory_allocated(device=None) / (1024*1024))

# from torch_cluster import knn as cluster_knn
# cindex = cluster_knn(data.double(), query.double(), k) # 2.3
# cindex = cindex[1].view(-1,20).transpose(1,0)

# print(cindex.shape)

# print(cindex[:, 1759])
# print(index[:, 1759])
#
# print(index[:,0])

#knn_points = data[index].contiguous().view(-1, 10, 3) # N, k, 3
# dist = torch.cdist(knn_points.double(), query.unsqueeze(1).double()).squeeze(-1) # N, k
# dist = dist.min(1)[0] # N
# for j in range(1, 10):
#     print(j, np.percentile(dist.detach().cpu().numpy(), j * 10))



# index = ball_query(query.unsqueeze(0), data.unsqueeze(0), K=20, return_nn=True)
# print(index[2].shape)

#index = knn(query, data, 20) # 1527.5654296875
#print(index.shape)
