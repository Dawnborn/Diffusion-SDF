#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate
from utils.reconstruct import *
from diff_utils.helpers import * 
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

from dataloader.pc_loader import PCloader, PCloaderDIT

from dataloader.sdf_loader import SdfLoader
from dataloader.modulation_loader import ModulationLoader

@torch.no_grad()
def test_modulations(recon_mesh=True):
    
    # load dataset, dataloader, model checkpoint
    test_split = json.load(open(specs["TestSplit"]))
    test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    ckpt = "{}.ckpt".format(args.resume) if args.resume == 'last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()

    # filename for logging chamfer distances of reconstructed meshes
    cd_file = os.path.join(recon_dir, args.resume, "cd.csv")

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            point_cloud, filename = data # filename = path to the csv file of sdf data
            filename = filename[0] # filename is a tuple

            cls_name = filename.split("/")[-2]
            mesh_name = filename.split("/")[-1]
            outdir = os.path.join(recon_dir, args.resume, "{}".format(cls_name))
            os.makedirs(outdir, exist_ok=True)
            mesh_filename = os.path.join(outdir, mesh_name)
            if os.path.exists(mesh_filename):
                print("mesh {} already exists!!!!!!".format(mesh_filename))
                continue
            else:
                print("processing: {}",format(mesh_filename))
            
            # given point cloud, create modulations (e.g. 1D latent vectors)
            plane_features = model.sdf_model.pointnet.get_plane_features(point_cloud.cuda())  # tuple, 3 items with ([1, D, resolution, resolution])
            plane_features = torch.cat(plane_features, dim=1) # ([1, D*3, resolution, resolution])
            recon = model.vae_model.generate(plane_features) # ([1, D*3, resolution, resolution]) [1,768,64,64]
            #print("mesh filename: ", mesh_filename)
            # N is the grid resolution for marching cubes; set max_batch to largest number gpu can hold
            mesh.create_mesh(model.sdf_model, recon, mesh_filename, N=256, max_batch=2**18, from_plane_features=True)

            # load the created mesh (mesh_filename), and compare with input point cloud
            # to calculate and log chamfer distance 
            mesh_log_name = cls_name+"/"+mesh_name
            try:
                evaluate.main(point_cloud, mesh_filename, cd_file, mesh_log_name)
            except Exception as e:
                print(e)


            # save modulation vectors for training diffusion model for next stage
            # filter based on the chamfer distance so that all training data for diffusion model is clean 
            # would recommend visualizing some reconstructed meshes and manually determining what chamfer distance threshold to use
            try:
                # skips modulations that have chamfer distance > 0.0018
                # the filter also weighs gaps / empty space higher
                cd = filter_threshold(mesh_filename, point_cloud)
                if cd > 0.0018:
                    print("mesh: {} {} larger than the threshold and will be skipped!!!".format(mesh_filename, cd))
                    continue
                outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)
                features = model.sdf_model.pointnet.get_plane_features(point_cloud.cuda())
                features = torch.cat(features, dim=1) # ([1, D*3, resolution, resolution])
                latent = model.vae_model.get_latent(features) # (1, D*3)
                np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
                print("latent is saved to {}".format(os.path.join(outdir, "latent.txt")))
            except Exception as e:
                print(e)


           
@torch.no_grad()
def test_generation():

    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # loads the sdf and vae models
            model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False) 

            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

            model = model.cuda().eval()
    else:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()

    conditional = specs["diffusion_model_specs"]["cond"] 

    if not conditional:
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
    else:
        # load dataset, dataloader, model checkpoint
        test_split = json.load(open(specs["TestSplit"]))
        test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

                point_cloud, filename = data # filename = path to the csv file of sdf data
                filename = filename[0] # filename is a tuple

                cls_name = filename.split("/")[-3]
                mesh_name = filename.split("/")[-2]
                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)

                # filter, set threshold manually after a few visualizations
                if args.filter:
                    threshold = 0.08
                    tmp_lst = []
                    count = 0
                    while len(tmp_lst)<args.num_samples:
                        count+=1
                        samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True) # batch should be set to max number GPU can hold
                        plane_features = model.vae_model.decode(samples)
                        # predicting the sdf values of the point cloud
                        perturbed_pc_pred = model.sdf_model.forward_with_plane_features(plane_features, perturbed_pc.repeat(args.num_samples, 1, 1))
                        consistency = F.l1_loss(perturbed_pc_pred, torch.zeros_like(perturbed_pc_pred), reduction='none')
                        loss = reduce(consistency, 'b ... -> b', 'mean', b = consistency.shape[0]) # one value per generated sample 
                        #print("consistency shape: ", consistency.shape, loss.shape, consistency[0].mean(), consistency[1].mean(), loss) # cons: [B,N]; loss: [B]
                        thresh_idx = loss<=threshold
                        tmp_lst.extend(plane_features[thresh_idx])

                        if count > 5: # repeat this filtering process as needed 
                            break
                    # skip the point cloud if cannot produce consistent samples or 
                    # just use the samples that are produced if comparing to other methods
                    if len(tmp_lst)<1: 
                        continue
                    plane_features = tmp_lst[0:min(10,len(tmp_lst))]

                else:
                    # for each point cloud, the partial pc and its conditional generations are all saved in the same directory 
                    samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True)
                    plane_features = model.vae_model.decode(samples)
                
                for i in range(len(plane_features)):
                    plane_feature = plane_features[i].unsqueeze(0)
                    mesh.create_mesh(model.sdf_model, plane_feature, outdir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
def test_generation_stage2(save_ptc=True):
    ckpt = "{}.ckpt".format(args.resume) if args.resume == 'last' else "epoch={}.ckpt".format(args.resume)
    print(ckpt)
    resume = os.path.join(args.exp_dir, ckpt)
    print(resume)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()
    print(model.specs['diffusion_model_specs']['cond'])
    test_split = json.load(open(specs["TestSplit"]))
    train_split = json.load(open(specs["TrainSplit"]))
    test_dataset = PCloader(specs["pc_path"], test_split, pc_size=specs.get("PCsize", 1024), return_filename=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

            point_cloud, filename = data  # filename = path to the csv file of sdf data
            filename = filename[0]  # filename is a tuple

            cls_name = filename.split("/")[-3]
            mesh_name = filename.split("/")[-2]
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            ptc_out_dir = outdir.replace("recon","ptc")
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(ptc_out_dir,exist_ok=True)

            samples, perturbed_pc, traj = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=1,
                                                                           save_pc=False, return_pc=True)
            # import pdb; pdb.set_trace()
            for i in range(samples.shape[0]):
                s = samples.cpu().numpy()
                # np.savetxt(s,os.path.join())
                print(s.shape)
                save_path = os.path.join(outdir, os.path.basename(filename)+".txt")
                print("save path:",save_path)
                np.savetxt(save_path, s)
                if save_ptc:
                    ptc_save_path = os.path.join(ptc_out_dir, os.path.basename(filename))
                    # save perturbed pc ply file for visualization
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
                    o3d.io.write_point_cloud(ptc_save_path+".pcd", pcd)

            # plane_features = model.vae_model.decode(samples)

            # for i in range(len(plane_features)):
            #     plane_feature = plane_features[i].unsqueeze(0)
            #     mesh.create_mesh(model.sdf_model, plane_feature, outdir + "/{}_recon".format(i), N=128,
            #                      max_batch=2 ** 21, from_plane_features=True)

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        # required=True,
        # default="config/stage1dit_sdf",
        default="config/repro_stage1_sdf",
        # default="config/repro_stage2_diff_cond",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well."
    )
    arg_parser.add_argument(
        "--resume", "-r",
        # default="last",
        default="9999",
        # default="4999",
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--num_samples", "-n", default=4, type=int, help='number of samples to generate and reconstruct')

    arg_parser.add_argument("--filter", default=False, help='whether to filter when sampling conditionally')

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])


    recon_dir = os.path.join(args.exp_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    
    if specs['training_task'] == 'modulation':
        latent_dir = os.path.join(args.exp_dir, "modulations", args.resume)
        os.makedirs(latent_dir, exist_ok=True)
        test_modulations()
    elif specs['training_task'] == 'combined':
        test_generation()
    elif specs['training_task'] == 'diffusion':
        print("test_generation_stage2")
        test_generation_stage2()

  
