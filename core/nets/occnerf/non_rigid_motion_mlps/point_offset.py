import torch
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.gridencoder import GridEncoder

class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 skips=None,
                 bound=1):
        super(NonRigidMotionMLP, self).__init__()

        self.skips = [1,2,3] if skips is None else skips
        self.bound = bound

        # for v9+
        self.encoder = GridEncoder(input_dim=3, num_levels=8, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048*bound, gridtype='hash', align_corners=False)
        # for v8
        #self.encoder = GridEncoder(input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048*bound, gridtype='hash', align_corners=False)
        block_mlps = [nn.Linear(self.encoder.output_dim+condition_code_size, 
                                mlp_width), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 3)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()


    def forward(self, xyz, condition_code, **_):

        h = self.encoder(xyz.view(-1, 3), bound=self.bound)
        
        h = torch.cat([condition_code, h], dim=-1)

        for i in range(len(self.block_mlps)):
            h = self.block_mlps[i](h)
        return h
