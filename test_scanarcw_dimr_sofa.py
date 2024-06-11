# from model.diffusion.model import *
from diff_utils.helpers import *

import os

from models import CombinedModel
from models.archs.deep_implicit_template_decoder import Decoder, load_SDF_model_from_specs, load_SDF_specs, remove_weight_norm, calc_and_fix_weights

from tqdm import tqdm

import pdb

from dataloader.dataset_ScanARCW import MyScanARCWDataset

import argparse

import numpy as np
import pytorch_lightning as pl
import random

import deep_sdf

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
                        # default="config/stage2_diff_cond_scanarcw",
                        # default="config/stage2_diff_cond_scanarcw_l1_10times42_nonperturb_b35",
                        # default="config/stage2_diff_cond_scanarcw_l1_1e-4_nonperturb_b70",
                        # default="config/stage2_diff_cond_scanarcw_4times420_b280",
                        # default="config/ddit_stage2_diff_cond",
                        # default="config/ddit_stage2_diff_cond_sofa_train_neighbor",
                        # default="config/ddit_stage2_diff_cond_sofa_train_noneighbor",
                        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_neighbor",
                        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_noneighbor",
                        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_neighbor",
                        default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/dimr_stage2_sofa",
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
                        # default="68999",
                        # default="23999",
                        # default="69999",
                        # default="49999",
                        # default="18999",
                        # default="27999",
                        # default="1999",
                        # default="9999",
                        default="1199",
                        help='Checkpoint number or "last".')

    parser.add_argument('--nocond',
                        default=False,
                        help='Flag to specify no condition mode.')

    parser.add_argument("--mode", default="test")

    parser.add_argument("--create_mesh", default=True)

    parser.add_argument("--max_batch", default=2 ** 17)

    parser.add_argument("--pre_load", default=False)

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
    diffusion_ckpt_path = os.path.join(config_path, "last.ckpt" if ckpt == "last" else "epoch={}.ckpt".format(ckpt))

    # output_path = "./output_lat/"
    # output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output"
    # output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw3/output/24999"
    # output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw7/output/499"
    if nocond:
        output_path = os.path.join(config_path, "output", ckpt, "nocond", args.mode, "lat")
        output_path_mesh = os.path.join(config_path, "output", ckpt, "nocond", args.mode, "mesh")
    else:
        output_path = os.path.join(config_path, "output", ckpt, args.mode, "lat")
        output_path_mesh = os.path.join(config_path, "output", ckpt, args.mode, "mesh")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_samples = 100
    num_samples = None

    specs = json.load(open(sepcs_path))
    print(specs["Description"])

    model = CombinedModel(specs)

    # model = model.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)

    ckpt = torch.load(diffusion_ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    model = model.cuda().eval()

    # dataset_train = MyScanARCWDataset(latent_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24/LatentCodes/train/2000/canonical_mesh_manifoldplus/04256520",
    #                            pcd_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA",
    #                            json_file_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/json_files_v5",
    #                            sdf_file_root="/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520",
    #                            pc_size=specs['diffusion_specs']['sample_pc_size'],
    #                            length=specs.get('dataset_length', -1),
    #                            pre_load=True,
    #                            use_sdf=False
    #                            )

    dataset_test = MyScanARCWDataset(
        latent_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24/LatentCodes/train/2000/canonical_mesh_manifoldplus/04256520",
        pcd_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA",
        json_file_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/json_files_v5",
        sdf_file_root="/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520",
        #    split_file=specs.get("TrainSplit",None),
        pc_size=specs['diffusion_specs'].get('sample_pc_size', 128),
        length=specs.get('dataset_length', -1),
        # length=60,
        times=specs.get('times', 1),
        pre_load=args.pre_load,
        conditional=specs["diffusion_model_specs"].get("cond", True),
        include_category=False,
        use_neighbor=specs.get('use_neighbor', False),
        #    preprocess="/storage/user/huju/transferred/ws_dditnach/DDIT/preprocess_output/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff_l1",
        preprocess=specs.get("preprocess", None),
        sdf_size=specs.get("sdf_samples", 20000),
        mode=args.mode,
        specs=specs,
        use_sdf=True,
        )

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1, num_workers=0
    )

    dataset_test.save_latent_paths(os.path.join(config_path, "test_list.txt"))
    # import pdb; pdb.set_trace()

    for idx, data in tqdm(enumerate(test_dataloader)):
        if num_samples:
            if idx >= num_samples:
                break
        pcd = data['point_cloud'].cuda().detach()
        # latent_gt = data['latent']
        latent_gt_path = data['latent_path'][0]
        # pdb.set_trace()

        # pcd = torch.from_numpy(pcd).to(torch.float32).cuda()

        latent_pred = model.generate_lat_from_pc_dimr(pcd).detach().cuda()

        lat_name = os.path.basename(latent_gt_path)
        lat_vec_path = os.path.join(output_path, lat_name)
        # pdb.set_trace()
        torch.save(latent_pred, lat_vec_path)

        ptc_out_dir = os.path.join(config_path, "pcd")
        os.makedirs(ptc_out_dir, exist_ok=True)
        ptc_save_path = os.path.join(ptc_out_dir, os.path.basename(lat_name) + ".pcd")
        if not os.path.exists(ptc_save_path):
            # save perturbed pc ply file for visualization
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd.cpu().numpy().squeeze())
            o3d.io.write_point_cloud(ptc_save_path, pcd_o3d)

        if args.create_mesh:

            decoder_specs, experiment_directory = load_SDF_specs(model.specs["ddit_specs"]["wanted_category"][0], model.specs["sdf_model"])
            model.decoder = load_SDF_model_from_specs(decoder_specs, experiment_directory)
            model.decoder.to(latent_pred.device)
            
            # mesh_dir = os.path.join(config_path, "mesh_test", args.mode, args.ckpt)
            if not os.path.isdir(output_path_mesh):
                os.makedirs(output_path_mesh)
            
            mesh_path = os.path.join( output_path_mesh, "{}.ply".format(lat_name.split(".")[0]) )
            if not os.path.exists(mesh_path):
                clamping_function = lambda x : torch.clamp(x, -0.1, 0.1)
                deep_sdf.mesh.create_mesh_octree(
                    model.decoder, latent_pred, mesh_path, N=256, max_batch=args.max_batch,
                    clamp_func=clamping_function
                )
                print("mesh saved to {}".format(mesh_path))
