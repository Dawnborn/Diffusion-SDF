import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
# knn = KNN(k=16, transpose_mode=False)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist  


class DGCNN_Grouper(nn.Module):
    def __init__(self, inputdim):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(inputdim, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, 3, n
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """


        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        x_dim = x.shape[1]
        coor_dim = coor.shape[1]

        new_coor = new_combined_x[:, :coor_dim]
        new_x = new_combined_x[:, coor_dim:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16 #! max sample number in local region
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
#             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M [B, 2048, nsample:16]
            idx = idx.transpose(-1, -2).contiguous() # [B, nsample:16, 2048]
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k # [B, 1, 1]
            idx = idx + idx_base # [B, nsample:16, 2048]
            idx = idx.view(-1)  # [B * nsample * 2048]
        num_dims = x_k.size(1) # feature dim 8 
        x_k = x_k.transpose(2, 1).contiguous() # B, np, 8
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :] # 重新排布 相近在一起
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous() # [B, num_dims, np, 16]
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k) # [B, num_dims, np, 16]
        feature = torch.cat((feature - x_q, x_q), dim=1) # [B, num_dims, np, 16] cat [B, num_dims, np, 16]
        return feature  # [B, 2*num_dims, np, 16] 前面是差值， 后面是local点

    def forward(self, x):

        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x # bs N(points number) 3 
        f = self.input_trans(x) # 1D conv -> bs N(points number) 8 

        f = self.get_graph_feature(coor, f, coor, f)  # 利用coor的信息，对f重新排列 —> [B, 2*num_dims:16, np, 16]
        f = self.layer1(f) #conv2d [B, 32, np, 16]
        f = f.max(dim=-1, keepdim=False)[0] # [B, 32, np]

        coor_q, f_q = self.fps_downsample(coor, f, 512) # 下采样到512，（相聚最远的512个）
        f = self.get_graph_feature(coor_q, f_q, coor, f) # [B, 64, 512, 16]
        f = self.layer2(f) # [B, 64, 512, 16]
        f = f.max(dim=-1, keepdim=False)[0] # [B, 64, 512]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f) # [B, 128, 512]
        f = self.layer3(f) # [B, 64, 512]
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, 128) # BOOKMARK: 整合成写死的128个组
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0] # [B, 128, 128]
        coor = coor_q # [B, 3, 128]

        return coor, f
