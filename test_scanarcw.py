#from model.diffusion.model import *
from diff_utils.helpers import * 

import os

from models import CombinedModel

from tqdm import tqdm

import pdb

from dataloader.dataset_ScanARCW import MyScanARCWDataset

if __name__ == "__main__":
    sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond_debug/specs.json"
    sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2/specs.json"
    sepcs_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/specs.json"
    
    diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond_test/last.ckpt"
    diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2/epoch=4999.ckpt"
    diffusion_ckpt_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/epoch=499.ckpt"
    
    output_path = "./output_lat/"
    output_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output"
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
                        pc_size=1024,
                        )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1, num_workers=0
    )


    for idx,data in tqdm(enumerate(train_dataloader)):
        if idx >= num_samples:
            break
        pcd = data['point_cloud']
        latent_gt = data['latent']
        latent_gt_path = data['latent_path'][0]
        # pdb.set_trace()

        # pcd = torch.from_numpy(pcd).to(torch.float32).cuda()
        pcd = pcd.cuda()
        samples = model.diffusion_model.generate_from_pc(pcd, batch=1)
        # pdb.set_trace()

        lat_vec = samples.squeeze()
        lat_name = os.path.basename(latent_gt_path)
        lat_vec_path = os.path.join(output_path, lat_name)
        # pdb.set_trace()
        torch.save(lat_vec, lat_vec_path)