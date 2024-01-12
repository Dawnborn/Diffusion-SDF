#!/usr/bin/env python3
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

import warnings

# add paths in model/__init__.py for new models
from utils.reconstruct import *
from diff_utils.helpers import *
# from metrics.evaluation_metrics import *#compute_all_metrics
# from metrics import evaluation_metrics

from dataloader.sdf_loader import SdfLoader, SdfLoaderDIT
from dataloader.modulation_loader import ModulationLoader

import pdb


def train():
    # initialize dataset and loader
    split = json.load(open(specs["TrainSplit"], "r"))
    if specs['training_task'] == 'diffusion':
        train_dataset = ModulationLoader(specs["data_path"], pc_path=specs.get("pc_path", None), split_file=split,
                                         pc_size=specs.get("total_pc_size", None))
    else:
        train_dataset = SdfLoaderDIT(specs["DataSource"], split_file=split, pc_size=specs.get("PCsize", 1024),
                                  grid_source=specs.get("GridSource", None),
                                  modulation_path=specs.get("modulation_path", None))

    print("======using a training dataset of length:{}".format(len(train_dataset)))
    # pdb.set_trace()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
    )

    # creates a copy of current code / files in the config folder
    save_code_to_conf(args.exp_dir)

    # pytorch lightning callbacks 
    callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{epoch}', save_top_k=-1, save_last=True,
                               every_n_epochs=specs["log_freq"])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [callback, lr_monitor]

    model = CombinedModel(specs)

    # note on loading from checkpoint:
    # if resuming from training modulation, diffusion, or end-to-end, just load saved checkpoint 
    # however, if fine-tuning end-to-end after training modulation and diffusion separately, will need to load sdf and diffusion checkpoints separately
    if args.resume == 'finetune':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k, v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.",
                                    "")  # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)
        resume = None
    elif args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume == 'last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        if not os.path.isfile(resume):
            print("ckpt not found!!!")
            resume = None
    else:
        resume = None

    # precision 16 can be unstable (nan loss); recommend using 32
    trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=specs["num_epochs"],
                         callbacks=callbacks, log_every_n_steps=1,
                         default_root_dir=os.path.join("tensorboard_logs", args.exp_dir))
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        # required=True,
        # default="config/stage2_diff_uncond2",
        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage1dit_sdf_grid",
        default="config/stage2_diff_uncond2_l1",
        # default=""
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default="last",
        # default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--batch_size", "-b", default=30, type=int)
    arg_parser.add_argument("--workers", "-w", default=8, type=int)
    arg_parser.add_argument("--debug", default=True, type=bool)

    args = arg_parser.parse_args()
    if args.debug:
        args.batch_size = 2
        args.workers = 1
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])

    train()