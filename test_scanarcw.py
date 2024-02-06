#from model.diffusion.model import *
from diff_utils.helpers import * 

import os

from models import CombinedModel

from tqdm import tqdm

import pdb

from dataloader.dataset_ScanARCW import MyScanARCWDataset

import argparse

import numpy as np
import pytorch_lightning as pl
import random

# 设置随机种子
seed = 42

# 设置随机种子用于PyTorch
pl.seed_everything(seed)

# 设置随机种子用于Numpy
np.random.seed(seed)

# 设置随机种子用于Python内置的random模块
random.seed(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run model with specified configuration and checkpoint.')

    parser.add_argument('--config_path', type=str,
                        # default="config/stage2_diff_cond_scanarcw_l1_pc1024",
                        # default="config/stage2_diff_cond_scanarcw_sinl1_pc1024_10times42",
                        # default="config/stage2_diff_cond_scanarcw_l1_pc1024_10times42",
                        # default="config/stage2_diff_cond_scanarcw_sinl1_pc1024_10times42_nonperturb",
                        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw",
                        # default="config/stage2_diff_cond_scanarcw_sinl1",
                        default="config/stage2_diff_cond_scanarcw",
                        # default="config/stage2_diff_cond_scanarcw_l1_10times42_nonperturb_b35",
                        # default="config/stage2_diff_cond_scanarcw_l1_1e-4_nonperturb_b70",
                        # default="config/stage2_diff_cond_scanarcw_4times420_b280",
                        help='Path to the configuration directory.')

    parser.add_argument('--ckpt', type=str,
                        # default="9999",
                        # default="11999",
                        # default="3599",
                        # default="12999",
                        # default="2999",
                        # default="16999",
                        # default="26999",
                        # default="27999",
                        # default="30999",
                        default="33999",
                        help='Checkpoint number or "last".')

    parser.add_argument('--nocond',
                        default=False,
                        help='Flag to specify no condition mode.')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    config_path = args.config_path
    ckpt = args.ckpt
    nocond = args.nocond
    # config_path = "/storage/user/lhao/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw9"
    # ckpt = "18199"

    # config_path = "/storage/user/lhao/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw10"
    # ckpt = "399"
    #
    # config_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw"
    # ckpt = "2499"
    # nocond=False

    # config_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw_l1_pc1024"
    # ckpt = "last"

    # sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond_debug/specs.json"
    # sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2/specs.json"
    # sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/specs.json"
    # sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw3/specs.json"
    # sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw7/specs.json"
    sepcs_path = os.path.join(config_path, "specs.json")

    # diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond_test/last.ckpt"
    # diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2/epoch=4999.ckpt"
    # diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/epoch=499.ckpt"
    # diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw3/epoch=24999.ckpt"
    # diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw7/epoch=499.ckpt"
    diffusion_ckpt_path = os.path.join(config_path, "last.ckpt" if ckpt=="last" else "epoch={}.ckpt".format(ckpt))


    # output_path = "./output_lat/"
    # output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output"
    # output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw3/output/24999"
    # output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw7/output/499"
    if nocond:
        output_path = os.path.join(config_path, "output", ckpt, "nocond")
    else:
        output_path = os.path.join(config_path, "output", ckpt)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
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

    dataset_train = MyScanARCWDataset(latent_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24/LatentCodes/train/2000/canonical_mesh_manifoldplus/04256520",
                               pcd_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA",
                               json_file_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/json_files_v5",
                               sdf_file_root="/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520",
                               pc_size=specs['diffusion_specs']['sample_pc_size'],
                               length=specs.get('dataset_length', -1),
                               pre_load=True,
                               )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1, num_workers=0
    )

    dataset_train.save_latent_paths(os.path.join(config_path, "train_list.txt"))
    # import pdb; pdb.set_trace()


    for idx,data in tqdm(enumerate(train_dataloader)):
        if idx >= num_samples:
            break
        pcd = data['point_cloud']
        latent_gt = data['latent']
        latent_gt_path = data['latent_path'][0]
        # pdb.set_trace()

        # pcd = torch.from_numpy(pcd).to(torch.float32).cuda()
        pcd = pcd.cuda()
        samples, perturbed_pc, traj= model.diffusion_model.generate_from_pc(pcd, batch=1, no_cond=nocond, return_pc=True)
        # pdb.set_trace()
        print("traj 0:", traj[0].mean(), traj[0].min(), traj[0].max())
        print("traj -1:",traj[-1].mean(), traj[-1].min(), traj[-1].max())
        print("gt:", latent_gt.mean(), latent_gt.min(), latent_gt.max())

        print("traj 0:", traj[0].mean(), traj[0].min(), traj[0].max())
        print("traj -1:",traj[-1].mean(), traj[-1].min(), traj[-1].max())
        print("gt:", latent_gt.mean(), latent_gt.min(), latent_gt.max())

        with torch.no_grad():
            e1 = torch.abs(traj[0].cpu() - latent_gt.cpu()).cpu()
            e2 = torch.abs(traj[-1].cpu() - latent_gt.cpu()).cpu()
            print("{} of digits reduced error".format((e1 > e2).float().mean()))

        lat_vec = samples.squeeze()
        lat_name = os.path.basename(latent_gt_path)
        lat_vec_path = os.path.join(output_path, lat_name)
        # pdb.set_trace()
        torch.save(lat_vec, lat_vec_path)

        ptc_out_dir = os.path.join(config_path,"pcd")
        os.makedirs(ptc_out_dir, exist_ok=True)
        ptc_save_path = os.path.join(ptc_out_dir, os.path.basename(lat_name)+".pcd")
        # save perturbed pc ply file for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
        o3d.io.write_point_cloud(ptc_save_path, pcd)