import torch

from lib.pointgroup_ops.functions import pointgroup_ops

import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict

class UBlock(torch.nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) 
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                torch.nn.LeakyReLU(0.01),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                torch.nn.LeakyReLU(0.01),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                if i == 0:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0] * 2, nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
                else:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))

            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(torch.nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            torch.nn.LeakyReLU(0.01),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            torch.nn.LeakyReLU(0.01),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class DIMR_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # m = cfg.m # 16 or 32
        m = 16

        input_c = 3
        norm_fn = functools.partial(torch.nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, 2*m, kernel_size=3, padding=1, bias=False, indice_key="subml")
        )

        block_reps = 2
        block = ResidualBlock
        self.unet = UBlock([2*m, 2*m, 4*m, 4*m, 6*m, 6*m, 8*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(2*m),
            torch.nn.LeakyReLU(0.01)
        )

        ### zs + score + bbox branch
        self.z_in = spconv.SparseSequential(
            spconv.SubMConv3d(2*m+3, 4*m, kernel_size=3, padding=1, bias=False, indice_key='z_subm1')
        )

        self.z_net = UBlock([4*m, 4*m, 6*m, 8*m], norm_fn, 2, block, indice_key_id=1)

        self.z_out = spconv.SparseSequential(
            norm_fn(4*m),
            torch.nn.LeakyReLU(0.01),
        )

        self.z_linear = torch.nn.Sequential(
            torch.nn.Linear(4*m+6, 8*m),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(8*m, 16*m),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(16*m, 32*m),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(32*m, 256), # zs
        )
        
    def clusters_voxelization(self, batch_idxs, feats, coords, fullscale, mode=4):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        # c_idxs = clusters_idx[:, 1].cuda().long()

        # clusters_points_feats = feats[c_idxs] # [sumNPoint, C]
        clusters_points_feats = feats
        # clusters_points_angles_label = angles[c_idxs, :12] # [sumNPoint, 12]
        # clusters_points_angles_residual = angles[c_idxs, 12:] # [sumNPoint, 12]
        # clusters_points_coords = coords[c_idxs] # [sumNPoint, 3]
        clusters_points_coords = coords
        # clusters_points_semantics = semantics[c_idxs] # [sumNPoint, 8]

        # # get the semantic label of each proposal
        # clusters_semantics = pointgroup_ops.sec_mean(clusters_points_semantics, clusters_offset.cuda()) # [nCluster, 8]

        # get mean angle as the bbox angle
        # clusters_points_angles_label = torch.softmax(clusters_points_angles_label, dim=1)
        # clusters_angles_label_mean = pointgroup_ops.sec_mean(clusters_points_angles_label, clusters_offset.cuda())  # (nCluster, 12), float
        # clusters_angles_residual_mean = pointgroup_ops.sec_mean(clusters_points_angles_residual, clusters_offset.cuda())  # (nCluster, 12), float

        # decode angles
        # clusters_angles_label_mean = torch.argmax(clusters_angles_label_mean, dim=1) # [nCluster, ] long
        # clusters_angles_residual_mean = torch.gather(clusters_angles_residual_mean * np.pi / 12, 1, clusters_angles_label_mean.unsqueeze(1)).squeeze(1)
        # detach !!!
        # clusters_angles = BBox.class2angle_cuda(clusters_angles_label_mean, clusters_angles_residual_mean).detach()

        # clusters_points_angles = torch.index_select(clusters_angles, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint,), float

        # clusters_coords_min_ori = pointgroup_ops.sec_min(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        # clusters_coords_max_ori = pointgroup_ops.sec_max(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        #clusters_coords_mean = pointgroup_ops.sec_mean(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        # clusters_centroid = (clusters_coords_max_ori + clusters_coords_min_ori) / 2 # (nCluster, 3), float

        # clusters_points_centroid = torch.index_select(clusters_centroid, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float

        
        # center coords
        # clusters_points_coords -= clusters_points_centroid

        # cos_ = torch.cos(-clusters_points_angles)
        # sin_ = torch.sin(-clusters_points_angles)
        
        # clusters_points_coords[:, 0], clusters_points_coords[:, 1] = clusters_points_coords[:, 0] * cos_ - clusters_points_coords[:, 1] * sin_, clusters_points_coords[:, 0] * sin_ + clusters_points_coords[:, 1] * cos_

        # concat canonical coords
        clusters_points_feats = torch.cat([clusters_points_feats, clusters_points_coords], dim=1)

        # clusters_coords_min = pointgroup_ops.sec_min(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        # clusters_coords_max = pointgroup_ops.sec_max(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        # clusters_bbox_size = clusters_coords_max - clusters_coords_min

        # clusters_scale = 1 / (clusters_bbox_size / fullscale).max(1)[0]  # (nCluster), float
        # clusters_scale = clusters_scale.unsqueeze(-1)
        
        # min_xyz = clusters_coords_min * clusters_scale  # (nCluster, 3), float
        # max_xyz = clusters_coords_max * clusters_scale
        
        # clusters_points_coords = clusters_points_coords * torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        # offset = - min_xyz

        
        # offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        # clusters_points_coords += offset

        # clusters_points_coords = clusters_points_coords.long()
        # clusters_points_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_points_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_points_coords, int(batch_idxs[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_points_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(batch_idxs[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def forward(self, data, training_mode='train'):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        '''
        B = data['B']
        input = data['input']
        input_map = data['input_map']
        coords = data['coords']
        batch_idxs = data['batch_idxs']

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        #### proposals voxelization again
        input_feats, inp_map = self.clusters_voxelization(batch_idxs, output_feats, coords, self.score_fullscale, mode=4)

        ### zs
        proposal_out = self.z_in(input_feats)
        proposal_out = self.z_net(proposal_out)
        proposal_out = self.z_out(proposal_out)
        proposal_zs = self.z_linear(proposal_out)
        
        ret = {}

        ret['proposal_zs'] = proposal_zs

        return ret