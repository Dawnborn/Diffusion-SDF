#!/usr/bin/env python3
import pdb
import time
import logging
import os
import random
import torch
import torch.utils.data
from . import base

import pandas as pd
import numpy as np
import csv, json
import trimesh

from tqdm import tqdm


class SdfLoader(base.Dataset):
    def __init__(
            self,
            data_source,  # path to points sampled around surface
            split_file,  # json.load(splitfile), which contains train/test classes and meshes
            grid_source=None,
            # path to grid points; grid refers to sampling throughout the unit cube instead of only around the surface; necessary for preventing artifacts in empty space
            samples_per_mesh=16000,
            pc_size=1024,
            modulation_path=None,
            suffix=".csv"
            # used for third stage of training; needs to be set in config file when some modulation training had been filtered
    ):
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.suffix=suffix
        self.gt_files = self.get_instance_filenames(data_source, split_file, filter_modulation_path=modulation_path,suffix=self.suffix)

        subsample = len(self.gt_files)
        self.gt_files = self.gt_files[0:subsample]

        self.grid_source = grid_source
        # print("grid source: ", grid_source)

        if grid_source:
            self.grid_files = self.get_instance_filenames(grid_source, split_file, gt_filename="grid_gt.csv",
                                                          filter_modulation_path=modulation_path,suffix=self.suffix)
            self.grid_files = self.grid_files[0:subsample]
            lst = []
            with tqdm(self.grid_files) as pbar:
                for i, f in enumerate(pbar):
                    pbar.set_description("Grid files loaded: {}/{}".format(i, len(self.grid_files)))
                    if not os.path.isfile(f):
                        import pdb
                        pdb.set_trace()
                    lst.append(torch.from_numpy(pd.read_csv(f, sep=',',comment='#').values))
            self.grid_files = lst

            assert len(self.grid_files) == len(self.gt_files)

        # load all csv files first 
        print("loading all {} files into memory...".format(len(self.gt_files)))
        lst = []
        with tqdm(self.gt_files) as pbar:
            for i, f in enumerate(pbar):
                pbar.set_description("Files loaded: {}/{}".format(i, len(self.gt_files)))
                lst.append(torch.from_numpy(pd.read_csv(f, sep=',', header=None, comment='#').values))
        self.gt_files = lst

    def __getitem__(self, idx):

        near_surface_count = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt = self.labeled_sampling(self.gt_files[idx], near_surface_count, self.pc_size,
                                                    load_from_path=False)

        if self.grid_source is not None:
            grid_count = self.samples_per_mesh - near_surface_count
            _, grid_xyz, grid_gt = self.labeled_sampling(self.grid_files[idx], grid_count, pc_size=0,
                                                         load_from_path=False)
            # each getitem is one batch so no batch dimension, only N, 3 for xyz or N for gt 
            # for 16000 points per batch, near surface is 11200, grid is 4800
            # print("shapes: ", pc.shape,  sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz))
            sdf_gt = torch.cat((sdf_gt, grid_gt))
            # print("shapes after adding grid: ", pc.shape, sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)

        data_dict = {
            "xyz": sdf_xyz.float().squeeze(),  # (B, N, 3)
            "gt_sdf": sdf_gt.float().squeeze(),  # (B, N)
            "point_cloud": pc.float().squeeze(),  # (B, 1024, 3)
        }

        return data_dict

    def __len__(self):
        return len(self.gt_files)

class SdfLoaderDIT(base.Dataset):
    def __init__(
            self,
            data_source,  # ./data/SdfSamples, path to points sampled around surface
            split_file,  # json filepath which contains train/test classes and meshes
            grid_source=None,
            surface_source=None,  # ./data/SurfaceSamples
            # path to grid points; grid refers to sampling throughout the unit cube instead of only around the surface; necessary for preventing artifacts in empty space
            samples_per_mesh=16000,
            pc_size=1024,
            modulation_path=None
            # used for third stage of training; needs to be set in config file when some modulation training had been filtered
    ):
        self.data_source = data_source

        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file, filter_modulation_path=modulation_path)

        subsample = len(self.gt_files)
        self.gt_files = self.gt_files[0:subsample]

        self.grid_source = grid_source

        if grid_source:
            self.grid_files = self.get_instance_filenames(grid_source, split_file, t="grid_data",
                                                          filter_modulation_path=modulation_path)
            self.grid_files = self.grid_files[0:subsample]
            lst = []
            with tqdm(self.grid_files) as pbar:
                for i, f in enumerate(pbar):
                    pbar.set_description("Grid files loaded: {}/{}".format(i, len(self.grid_files)))
                    lst.append(torch.from_numpy(np.loadtxt(f,comments='#', delimiter=",")))
            self.grid_files = lst

            assert len(self.grid_files) == len(self.gt_files)

        # load all csv files first
        # print("loading all {} files into memory...".format(len(self.gt_files)))
        # lst = []
        # with tqdm(self.gt_files) as pbar:
        #     for i, f in enumerate(pbar):
        #         pbar.set_description("Files loaded: {}/{}".format(i, len(self.gt_files)))
        #         lst.append(torch.from_numpy(pd.read_csv(f, sep=',', header=None).values))
        # self.gt_files = lst

    def get_instance_filenames(self, data_source, split_json, filter_modulation_path=None, t="SdfSamples"):
        """

        :param data_source: /home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepSDF/data
        :param split_json: dict from .json file
        :param filter_modulation_path:
        :return:
        """
        if t=="SdfSamples":
            suffix = ".npz"
        elif t=="grid_data":
            suffix = ".csv"
        else:
            import pdb
            pdb.set_trace()

        do_filter = filter_modulation_path is not None
        ret_files = []
        for dataset in split_json:
            for category_id in split_json[dataset]:
                for instance_file in split_json[dataset][category_id]:
                    instance_file_path = os.path.join(data_source, t, dataset, category_id, instance_file+suffix)
                    if not os.path.isfile(instance_file_path):
                        logging.warning("Requested non-existent file '{}'".format(instance_file_path))
                        import pdb
                        pdb.set_trace()
                    else:
                        ret_files.append(instance_file_path)

        return ret_files

    def labeled_sampling(self, f_path, subsample, pc_size=1024, load_from_path=True):

        assert (os.path.basename(f_path).split(".")[-1] == "npz")
        if not os.path.exists(f_path):
            print("f_path:{} not exists!!!".format(f_path))
            import pdb
            pdb.set_trace()
        f = np.load(f_path)
        neg_tensor = f["neg"]
        neg_tensor = neg_tensor[~np.isnan(neg_tensor[:, 3])]
        pos_tensor = f["pos"]
        pos_tensor = pos_tensor[~np.isnan(pos_tensor[:,3])]

        half = int(subsample / 2)

        if pos_tensor.shape[0] > half:
            pos_idx = np.random.permutation(pos_tensor.shape[0])[:half]
        else:
            pos_idx = np.random.randint(0, pos_tensor.shape[0], half)
        pos_tensor = pos_tensor[pos_idx]

        if neg_tensor.shape[0] > half:
            neg_idx = np.random.permutation(neg_tensor.shape[0])[:half]
        else:
            neg_idx = np.random.randint(0, neg_tensor.shape[0], half)
        neg_tensor = neg_tensor[neg_idx]

        samples = np.vstack((pos_tensor, neg_tensor))
        assert (samples.shape[0] == subsample)

        mesh_path = f_path.replace("SdfSamples", "SurfaceSamples").replace(".npz", ".ply")
        if not os.path.exists(mesh_path):
            import pdb
            pdb.set_trace()

        pc = np.array(trimesh.load(mesh_path).vertices)
        pc_idx = np.random.permutation(pc.shape[0])[:pc_size]
        pc = pc[pc_idx]

        translation = np.array(f["translation_mesh2sdf"])
        scale = np.array(f["scale_mesh2sdf"])
        pc = (pc + translation) * scale

        # return pc.float().squeeze(), samples[:, :3].float().squeeze(), samples[:, 3].float().squeeze()  # pc, xyz, sdv
        return pc, samples[:, :3], samples[:, 3]

    def __getitem__(self, idx):
        near_surface_count = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh
        pc, sdf_xyz, sdf_gt = self.labeled_sampling(self.gt_files[idx], subsample=near_surface_count, pc_size=self.pc_size)
        if self.grid_source is not None:
            grid_count = self.samples_per_mesh - near_surface_count

        if np.isnan(sdf_gt).any():
            print("nan in file:{}".format(self.gt_files[idx]))
            # import pdb
            # pdb.set_trace()

        data_dict = {
            "xyz": sdf_xyz.astype(np.float32),
            "gt_sdf": sdf_gt.astype(np.float32),
            "point_cloud": pc.astype(np.float32),
        }
        return data_dict
    def __len__(self):
        return len(self.gt_files)

if __name__ == "__main__":
    specs = json.load("/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/mystage1_sdf/specs.json")
    specs = json.load("/storage/user/lhao/hjp/ws_dditnach/Diffusion-SDF/config/stage1dit_sdf/specs.json")
    split = json.load(open(specs["TrainSplit"], "r"))
    train_dataset = SdfLoaderDIT(specs["DataSource"], split, pc_size=specs.get("PCsize",1024), grid_source=specs.get("GridSource", None), modulation_path=specs.get("modulation_path", None))
    pass
