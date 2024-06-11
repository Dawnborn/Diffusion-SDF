#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

import os
import json

from collections import OrderedDict

import torch.nn.utils as utils

def remove_weight_norm(model):
    for name, module in model.named_children():
        if hasattr(module, 'weight_g'):
            utils.remove_weight_norm(module)
        remove_weight_norm(module)  # 递归处理子模块

def calc_and_fix_weights(model):
    for module in model.modules():
        if hasattr(module, 'weight_v') and hasattr(module, 'weight_g'):
            w = module.weight_g * module.weight_v / torch.norm(module.weight_v, dim=1, keepdim=True)
            module.weight = torch.nn.Parameter(w)  # 转换成普通参数
            del module._parameters['weight_g']  # 删除不再需要的属性
            del module._parameters['weight_v']

class SdfDecoderold(nn.Module):
    def __init__(
        self,
        dims, # [256, 256, 256, 256, 256]
        dropout=None, # [0, 1, 2, 3, 4]
        dropout_prob=0.0, # 0.05
        norm_layers=(), # [0,1,2,3,4]
        xyz_in_all=None, # False
        use_tanh=False, # False
        weight_norm=False, # True
    ):
        super(SdfDecoder, self).__init__()
        dims = [3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            if self.xyz_in_all and layer != self.num_layers - 2:
                out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

class SdfDecodernew(nn.Module):
    def __init__(self, dims, dropout=None, dropout_prob=0.0, norm_layers=(), xyz_in_all=None, use_tanh=False, weight_norm=False):
        super(SdfDecodernew, self).__init__()
        self.layers = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.use_tanh = use_tanh
        self.xyz_in_all = xyz_in_all
        self.relu = nn.ReLU()
        
        input_dim = 3  # Initial input dimension for xyz coordinates
        dims = [input_dim] + dims + [1]  # Append output dimension

        for i in range(len(dims) - 1):
            out_dim = dims[i + 1]
            if xyz_in_all and i != len(dims) - 2:
                out_dim -= 3  # Adjust dimensions if xyz coordinates are included in all but the last layer

            # Create linear layers with optional weight normalization
            layer_name = f"lin{i}"
            if weight_norm and i in norm_layers:
                self.layers[layer_name] = nn.utils.weight_norm(nn.Linear(dims[i], out_dim))
            else:
                self.layers[layer_name] = nn.Linear(dims[i], out_dim)

            # Add optional batch normalization layers
            if i in norm_layers and not weight_norm:
                self.norms[f"bn{i}"] = nn.LayerNorm(out_dim)

            # Dropout is handled during the forward pass

        # Optional tanh activation for the output layer
        if use_tanh:
            self.tanh = nn.Tanh()
        
        self.dropout = dropout
        self.dropout_prob = dropout_prob

    def forward(self, input):
        x = input
        xyz = input[:, -3:] if self.xyz_in_all else None

        for i, (name, lin) in enumerate(self.layers.items()):
            if self.xyz_in_all and i != len(self.layers) - 1:
                x = torch.cat([x, xyz], dim=1)
            
            x = lin(x)

            if name in self.norms:
                x = self.norms[name](x)

            if i < len(self.layers) - 1:  # Apply ReLU on all but the last layer
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
            if i == len(self.layers) - 1 and self.use_tanh:
                x = self.tanh(x)

        return x
    
class SdfDecodernew2(nn.Module):
    def __init__(
        self,
        dims, # [256, 256, 256, 256, 256]
        dropout=None, # [0, 1, 2, 3, 4]
        dropout_prob=0.0, # 0.05
        norm_layers=(), # [0,1,2,3,4]
        xyz_in_all=None, # False
        use_tanh=False, # False
        weight_norm=False, # True
    ):
        super(SdfDecoder, self).__init__()
        self.use_tanh = use_tanh
        modules = OrderedDict()

        input_dim = 3  # Initial input dimension for xyz coordinates
        dims = [input_dim] + dims + [1]  # Append output dimension
        
        for i in range(len(dims) - 1):
            out_dim = dims[i + 1]
            layer_name = f"lin{i}"

            # Apply weight normalization if specified and in norm_layers
            if weight_norm and i in norm_layers:
                modules[layer_name] = nn.utils.weight_norm(nn.Linear(dims[i], out_dim), name='weight')
            else:
                modules[layer_name] = nn.Linear(dims[i], out_dim)

            # Add normalization layers if specified and not using weight normalization
            if i in norm_layers and not weight_norm:
                norm_name = f"bn{i}"
                modules[norm_name] = nn.LayerNorm(out_dim)

            # Add ReLU activation for all but the last layer
            if i < len(dims) - 2:
                modules[f"relu{i}"] = nn.ReLU()
                # Add dropout if specified
                if dropout_prob > 0:
                    modules[f"dropout{i}"] = nn.Dropout(dropout_prob)

        # Optionally add a Tanh activation for the output layer
        if use_tanh:
            modules["tanh"] = nn.Tanh()

        # Create the sequential model
        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)

class SdfDecoder2(nn.Module):
    def __init__(
        self,
        dims, # [256, 256, 256, 256, 256]
        dropout=None, # [0, 1, 2, 3, 4]
        dropout_prob=0.0, # 0.05
        norm_layers=(), # [0,1,2,3,4]
        xyz_in_all=None, # False
        use_tanh=False, # False
        weight_norm=False, # True
    ):
        super(SdfDecoder, self).__init__()
        # 固定参数
        dropout_prob = 0.05
        use_tanh = False
        weight_norm = True

        # 定义每一层
        if weight_norm:
            self.lin0 = nn.utils.weight_norm(nn.Linear(3, 256))
            self.lin1 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin2 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin3 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin4 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin5 = nn.Linear(256, 1)
        else:
            self.lin0 = nn.Linear(3, 256)
            self.lin1 = nn.Linear(256, 256)
            self.lin2 = nn.Linear(256, 256)
            self.lin3 = nn.Linear(256, 256)
            self.lin4 = nn.Linear(256, 256)
            self.lin5 = nn.Linear(256, 1)

        # 所有层使用相同的标准化
        self.bn0 = nn.LayerNorm(256)
        self.bn1 = nn.LayerNorm(256)
        self.bn2 = nn.LayerNorm(256)
        self.bn3 = nn.LayerNorm(256)
        self.bn4 = nn.LayerNorm(256)
        self.bn5 = nn.LayerNorm(1)

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        if use_tanh:
            self.tanh = nn.Tanh()

    def forward(self, input):
        x = self.relu(self.bn0(self.lin0(input)))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.bn1(self.lin1(x)))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.bn2(self.lin2(x)))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.bn3(self.lin3(x)))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.bn4(self.lin4(x)))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.bn5(self.lin5(x))
        if hasattr(self, 'tanh'):
            x = self.tanh(x)
        return x
    
class SdfDecoder(nn.Module):
    def __init__(
        self,
        dims, # [256, 256, 256, 256, 256]
        dropout=None, # [0, 1, 2, 3, 4]
        dropout_prob=0.0, # 0.05
        norm_layers=(), # [0,1,2,3,4]
        xyz_in_all=None, # False
        use_tanh=False, # False
        weight_norm=False, # True
    ):
        super(SdfDecoder, self).__init__()
        # 固定参数
        dropout_prob = 0.05
        use_tanh = False
        weight_norm = True

        # 定义每一层
        if weight_norm:
            self.lin0 = nn.utils.weight_norm(nn.Linear(3, 256))
            self.lin1 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin2 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin3 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin4 = nn.utils.weight_norm(nn.Linear(256, 256))
            self.lin5 = nn.Linear(256, 1)
        else:
            self.lin0 = nn.Linear(3, 256)
            self.lin1 = nn.Linear(256, 256)
            self.lin2 = nn.Linear(256, 256)
            self.lin3 = nn.Linear(256, 256)
            self.lin4 = nn.Linear(256, 256)
            self.lin5 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob

    def forward(self, input):
        x = self.relu(self.lin0(input))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.lin3(x))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.lin4(x))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = (self.lin5(x))

        return x

def remove_module_prefix(state_dict,prefix="module."):
    """移除state_dict中的`module.`前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(prefix, '')  # 将'module.'替换为空
        new_state_dict[new_key] = value
    return new_state_dict



def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

class SimpleWarper(nn.Module): # only for debug
    def __init__(self):
        super(SimpleWarper, self).__init__()
        # 定义一个 1D 卷积层，输入通道数和输出通道数都是3，卷积核大小为1
        # 这意味着卷积操作将在每个点独立进行，没有跨点的操作
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1)
    
    def forward(self, x):
        # x 的形状应该是 (B, N, 3)
        # 转换为 Conv1d 需要的输入形状 (B, 3, N)
        x = x.transpose(1, 2)
        
        # 应用 1D 卷积
        x = self.conv1d(x)
        
        # 将输出的形状恢复为 (B, N, 3)
        x = x.transpose(1, 2)
        
        return x

class Warper(nn.Module):
    def __init__(
            self,
            latent_size,
            hidden_size,
            steps,
    ):
        super(Warper, self).__init__()
        self.n_feature_channels = latent_size + 3
        self.steps = steps
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)
        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer_coord_affine = nn.Linear(hidden_size, 6)
        self.out_layer_coord_affine.apply(init_out_weights)

    def forward(self, input, step=1.0):
        if step < 1.0:
            input_bk = input.clone().detach()

        xyz = input[:, -3:]
        code = input[:, :-3]
        states = [None]
        warping_param = []

        warped_xyzs = []
        for s in range(self.steps):
            state = self.lstm(torch.cat([code, xyz], dim=1).cuda(), states[-1])
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))
            a = self.out_layer_coord_affine(state[0])
            tmp_xyz = torch.addcmul(a[:, 3:], (1 + a[:, :3]), xyz)

            warping_param.append(a)
            states.append(state)
            if (s+1) % (self.steps // 4) == 0:
                warped_xyzs.append(tmp_xyz)
            xyz = tmp_xyz

        if step < 1.0:
            xyz_ = input_bk[:, -3:]
            xyz = xyz * step + xyz_ * (1 - step)

        return xyz, warping_param, warped_xyzs


class Decoder(nn.Module):
    def __init__(self, latent_size, warper_kargs, decoder_kargs):
        super(Decoder, self).__init__()
        self.warper = Warper(latent_size, **warper_kargs)
        # self.warper = SimpleWarper()
        self.sdf_decoder = SdfDecoder(**decoder_kargs)
        # self.sdf_decoder = nn.Linear(3,1)

    def forward(self, input, output_warped_points=False, output_warping_param=False,
                step=1.0):
        """
        return p_final, x, warping_param
        return p_final, x
        """
        # pass
        print(input.size())
        print(input[0, :])

        p_final, _, _ = self.warper.forward(input, step=step) # tensor.size([40000,3]), list of 8 torch.Size([40000, 6]), 
        
        if not self.training:
            x = self.sdf_decoder(p_final)
            if output_warped_points:
                if output_warping_param:
                    return p_final, x, _
                else:
                    return p_final, x
            else:
                if output_warping_param:
                    return x, _
                else:
                    return x
        else:   # training mode, output intermediate positions and their corresponding sdf prediction
            xs=self.sdf_decoder(p_final)
            if output_warped_points:
                if output_warping_param:
                    return xs # [40000,1]
                else:
                    return _, xs
            else:
                if output_warping_param:
                    return xs, _
                else:
                    return xs

    def forward_template(self, input):
        return self.sdf_decoder(input)

def load_SDF_specs(categ_id, sdf_model_folder="pretrained"):
    #TODO 不写死
    if categ_id == '03001627':
        experiment_directory = os.path.join(sdf_model_folder, "chairs_dit")
    elif categ_id == '04256520':
        experiment_directory = os.path.join(sdf_model_folder, "sofas_dit")
    elif categ_id == '04379243':
        experiment_directory = os.path.join(sdf_model_folder, "tables_dit")
    elif categ_id == '02871439':
        experiment_directory = os.path.join(sdf_model_folder, "bookshelfs_dit")
    elif categ_id == '02818832':
        experiment_directory = os.path.join(sdf_model_folder, "beds_dit")
    elif categ_id == '02808440':
        experiment_directory = os.path.join(sdf_model_folder, "bathtubs_dit")
    elif categ_id == '02933112':
        experiment_directory = os.path.join(sdf_model_folder, "cabinets_dit")
    else:
        raise ValueError('Unknown categ_id')
    specs_filename = os.path.join(experiment_directory, "specs.json")
    with open(specs_filename, 'r') as f:
        specs = json.load(open(specs_filename))
    
    return specs, experiment_directory


def load_SDF_model_from_specs(decoder_specs, experiment_directory, checkpoint = "latest") -> Decoder:
    decoder = Decoder(decoder_specs["CodeLength"], **decoder_specs["NetworkSpecs"])
    print(f"load Pretrained SDF model in: {experiment_directory} (Ignore this message when inference.)")
    saved_model_state = torch.load(
                os.path.join(experiment_directory, "ModelParameters", checkpoint + ".pth"),\
                )
    # decoder.to(device)
    # decoder = torch.nn.DataParallel(decoder, device_ids=parallerl_device)
    state_dict = remove_module_prefix(saved_model_state["model_state_dict"])
    decoder.load_state_dict(state_dict)
    return decoder
    

# def load_decoder(categ_id, sdf_model_folder, device, parallerl_device) -> Decoder: 
#     specs, experiment_directory = load_SDF_specs(categ_id, sdf_model_folder)
#     return load_SDF_from_specs(specs, experiment_directory, device, parallerl_device)