from typing import Tuple

import torch
from torch import Tensor
from pykeops.torch import LazyTensor


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
    
    return ranges, slices


def diagonal_ranges(batch_x = None, batch_y = None):
    """Encodes the block-diagonal structure associated to a batch vector."""
    
    if batch_y is None: return None
    
    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)
    print(ranges_x, slices_x, ranges_x.shape, slices_x.shape)
    print(ranges_y, slices_y)
    #ranges_x = torch.tensor([[0,q_size]]).view(1,2).to(ranges_y)
    #slices_x = torch.tensor([1]).view(1,).to(slices_y)
    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x 


def keops_knn(q_points, s_points, k, ranges_x=None, slices_x=None, ranges_y=None, slices_y=None) -> Tuple[Tensor, Tensor]:
    """kNN with PyKeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)

    if ranges_y is None:
        if type(s_points) is not list:
            xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
            dij = (xi - xj).norm2()  # (*, N, M)

            knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)

            return knn_distances.contiguous(), knn_indices.contiguous().detach()
        else:
            if type(s_points[0]) is LazyTensor:
                return None, [(xi - s_point).norm2().Kmin_argKmin(k, dim=num_batch_dims + 1)[1].contiguous() for s_point in s_points]
            else:
                return None, [(xi - LazyTensor(s_point.to(q_points).unsqueeze(-3))).norm2().Kmin_argKmin(k, dim=num_batch_dims + 1)[1].contiguous().detach() for s_point in s_points]
            
            # for s_point in s_points:
            #     #xj = LazyTensor(s_point.to(q_points).unsqueeze(-3))  # (*, 1, M, C)
            #     dij = (xi - s_point).norm2()

    elif ranges_x is None:
        indices = []
        #for s_point in s_points:
            #xj = LazyTensor(s_point.to(q_points).unsqueeze(-3))  # (*, 1, M, C)
        dij = (xi - s_points).norm2()  # (*, N, M)
        # print(dij.shape, dij.ndim)
        # print(ranges_y)
        # print(dij[10:20].shape)

        #indices.append(dij.Kmin_argKmin(k, dim=num_batch_dims + 1)[1].contiguous())  # (*, N, K)
        return None, [dij[indices[0]:indices[1]].Kmin_argKmin(k, dim=num_batch_dims + 1)[1].contiguous() for indices in ranges_y]
    else:
        xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
        dij = (xi - xj).norm2()  # (*, N, M)
        #print(dij.axis)
        dij.ranges = (ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x) #diagonal_ranges(batch_x, batch_y)

        _, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1) 
        #print(knn_indices.shape)
        return None, knn_indices
    #print(dij.shape)
    #dists = []
    #return_idx = []
    # knn_indices = dij.argKmin(k, dim=1)  # (B*N, K)
    # knn_distances = knn_indices
    # knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)
    # #if index is None:
    #     #knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)
    # return knn_distances.contiguous(), knn_indices.contiguous()
    
    
    #else:
    #dij = dij.dense()
    #    return None, [knn_indices] + [LazyTensor(dij[:,idx]).Kmin_argKmin(k, dim=num_batch_dims + 1)[1].contiguous() for idx in index]


def knn(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = False,
    remove_nearest: bool = False,
    transposed: bool = False,
    padding_mode: str = "nearest",
    inf: float = 1e10,
    ranges_x=None, 
    slices_x=None, 
    ranges_y=None, 
    slices_y=None
):
    """
    Compute the kNNs of the points in `q_points` from the points in `s_points`.

    Use KeOps to accelerate computation.

    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are replaced according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): padding mode for neighbors further than distance radius. ('nearest', 'empty').
        inf (float=1e10): infinity value for padding.

    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)

    #num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = dilated_k #min(dilated_k, num_s_points)

    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k, ranges_x, slices_x, ranges_y, slices_y)  # (*, N, k)
    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    #knn_distances = knn_distances.contiguous()
    #knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances[knn_masks] = knn_distances[..., 0]
            knn_indices[knn_masks] = knn_indices[..., 0]
        else:
            knn_distances[knn_masks] = inf
            knn_indices[knn_masks] = num_s_points

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices