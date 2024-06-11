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


from dataloader.dataset_ScanARCW import MyScanARCWDataset

# 设置随机种子
seed = 42

# 设置随机种子用于PyTorch
pl.seed_everything(seed)

# 设置随机种子用于Numpy
np.random.seed(seed)

# 设置随机种子用于Python内置的random模块
random.seed(seed)

def train():

    dataset_train = MyScanARCWDataset(latent_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/bookshelfs_dit_manifoldplus_scanarcw_origprep_all_large_pretrainedsofas/LatentCodes/train/2000/canonical_mesh_manifoldplus/02871439",
                               pcd_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA",
                               json_file_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/json_files_v5",
                               sdf_file_root="/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/02871439",
                            #    split_file=specs.get("TrainSplit",None),
                               pc_size=specs['diffusion_specs'].get('sample_pc_size', 128),
                               length=specs.get('dataset_length', -1),
                            #    length=10,
                               times=specs.get('times', 1),
                               pre_load=args.pre_load,
                               conditional=specs["diffusion_model_specs"].get("cond", True),
                               include_category=False,
                               use_neighbor=specs.get('use_neighbor', False),
                            #    preprocess="/storage/user/huju/transferred/ws_dditnach/DDIT/preprocess_output/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff_l1",
                               preprocess=specs.get("preprocess", None),
                               sdf_size=specs.get("sdf_samples",20000),
                               use_sdf=True,
                               mode="train",
                               specs=specs
                               )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
    )

    # creates a copy of current code / files in the config folder
    save_code_to_conf(args.exp_dir)

    print(args.exp_dir)
    
    # pytorch lightning callbacks 
    callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{epoch}', save_top_k=-1, save_last=True, every_n_epochs=specs["log_freq"])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [callback, lr_monitor]

    model = CombinedModel(specs, dataloader=train_dataloader)

    resume=None
    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        if not os.path.isfile(resume):
            print("ckpt not found!!!")
            resume=None

    trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=specs["num_epochs"], callbacks=callbacks, log_every_n_steps=1,
                        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir))
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)



if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", 
        # default="config/ddit_stage2_diff_cond_chair_train_neighbor",
        # default="config/ddit_stage2_diff_cond_chair_train_noneighbor",
        default="config/ddit_stage2_diff_cond_bookshelf_train_noneighbor",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", 
        default="last",
        # default="29999",
        # default="26999",
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument(
        "--end2end", 
        # default="last",
        default=False,
        help="end to end supervision",
    )

    arg_parser.add_argument("--batch_size", "-b", default=10, type=int)
    arg_parser.add_argument("--workers", "-w", default=12, type=int)
    arg_parser.add_argument("--pre_load", "-p", default=False, type=bool)
    # arg_parser.add_argument("--pre_load", "-p", action='store_true')

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])

    train()