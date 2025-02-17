import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

#from model.diffusion.model import * 
from diff_utils.helpers import * 

import numpy as np
import os
from statistics import mean
from tqdm.auto import tqdm
import open3d as o3d

from models import CombinedModel

from tqdm import tqdm

import pdb

if __name__ == "__main__":
    sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond_debug/specs.json"
    diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond_test/last.ckpt"
    output_path = "./output_lat/"
    num_samples = 100

    specs = json.load(open(sepcs_path))
    print(specs["Description"])

    model = CombinedModel(specs)

    # model = model.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)

    ckpt = torch.load(diffusion_ckpt_path)
    new_state_dict = {}
    for k,v in ckpt['state_dict'].items():
        new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
        new_state_dict[new_key] = v
    model.diffusion_model.load_state_dict(new_state_dict)

    model = model.cuda().eval()

    samples = model.diffusion_model.generate_unconditional(num_samples)

    for idx, lat_vec in tqdm(enumerate(samples)):
        lat_vec_path = os.path.join(output_path,"test_{}.pth".format(idx))
        torch.save(lat_vec, lat_vec_path)

    # pdb.set_trace()