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

from dataloader.sdf_loader import SdfLoader
from dataloader.modulation_loader import ModulationLoader
from dataloader.dataset_ScanARCW import MyScanARCWDataset


def train():

    dataset_train = MyScanARCWDataset(latent_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24/LatentCodes/train/2000/canonical_mesh_manifoldplus/04256520",
                               pcd_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA",
                               json_file_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/json_files_v5",
                               sdf_file_root="/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520",
                               pc_size=1024
                               )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
    )

    # creates a copy of current code / files in the config folder
    save_code_to_conf(args.exp_dir) 
    
    # pytorch lightning callbacks 
    callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{epoch}', save_top_k=-1, save_last=True, every_n_epochs=specs["log_freq"])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [callback, lr_monitor]

    model = CombinedModel(specs)

    resume=None
    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        if not os.path.isfile(resume):
            print("ckpt not found!!!")

    trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=specs["num_epochs"], callbacks=callbacks, log_every_n_steps=1,
                        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir))
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)



if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", 
        # required=True,
        # default = "config/stage2_diff_uncond_debug",
        # default = "config/stage2_diff_cond_debug",
        # default = "config/stage2_diff_uncond_debug",
        # default="config/stage1dit_sdf_grid",
        # default="config/stage2_diff_uncond2_l1",
        # default="config/repro_stage1_sdf",
        # default="config/repro_stage2_diff_cond",
        default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", 
        # default="last",
        default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--batch_size", "-b", default=70, type=int)
    arg_parser.add_argument("--workers", "-w", default=12, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])

    train()