#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 
from tqdm import tqdm

import pandas as pd 
import csv

import trimesh
import numpy as np

class PCloader(base.Dataset):

    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        pc_size=1024,
        return_filename=False
    ):

        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file)
        self.return_filename = return_filename

        self.pc_paths = self.get_instance_filenames(data_source, split_file)
        # self.pc_paths = self.pc_paths[:5]
        # print("loading {} point clouds into memory...".format(len(self.pc_paths)))
        # lst = []
        # with tqdm(self.pc_paths) as pbar:
        #     for i, f in enumerate(pbar):
        #         pbar.set_description("Files loaded: {}/{}".format(i, len(self.pc_paths)))
        #         lst.append(self.sample_pc(f, pc_size))
        # self.point_clouds = lst

        #print("each pc shape: ", self.point_clouds[0].shape)

    def get_all_files(self):
        return self.point_clouds, self.pc_paths 
    
    def __getitem__(self, idx):
        pc = self.sample_pc(self.pc_paths[idx], self.pc_size)
        if self.return_filename:
            return pc, self.pc_paths[idx]
        else:
            return pc


    def __len__(self):
        return len(self.pc_paths)


    def sample_pc(self, f, samp=1024): 
        '''
        f: path to csv file
        '''
        # data = torch.from_numpy(np.loadtxt(f, delimiter=',')).float()
        data = torch.from_numpy(pd.read_csv(f, sep=',', header=None, comment="#").values).float()
        pc = data[data[:,-1]==0][:,:3]
        pc_idx = torch.randperm(pc.shape[0])[:samp] 
        pc = pc[pc_idx]
        #print("pc shape, dtype: ", pc.shape, pc.dtype) # [1024,3], torch.float32
        #pc = normalize_pc(pc)
        #print("pc shape: ", pc.shape, pc.max(), pc.min())
        return pc


class PCloaderDIT(base.Dataset):

    def __init__(
            self,
            data_source,
            split_file,  # json filepath which contains train/test classes and meshes
            pc_size=1024,
            return_filename=False,
            num_subsample=None
    ):

        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file, t="SurfaceSamples", suffix=".ply")
        self.return_filename = return_filename

        self.sdf_files = self.get_instance_filenames(data_source, split_file, t="SdfSamples", suffix=".npz")
        self.norm_files = self.get_instance_filenames(data_source, split_file, t="NormalizationParameters", suffix=".npz")

        self.pc_paths = self.get_instance_filenames(data_source, split_file, t="SurfaceSamples", suffix=".ply")
        if num_subsample:
            self.pc_paths = self.pc_paths[:num_subsample]
        print("loading {} point clouds into memory...".format(len(self.pc_paths)))
        lst = []
        # with tqdm(self.pc_paths) as pbar:
        #     for i, f in enumerate(pbar):
        #         pbar.set_description("Files loaded: {}/{}".format(i, len(self.pc_paths)))
        #         pc = self.sample_pc(f, pc_size)
        #
        #         norm_file = self.norm_files[i]
        #         if not os.path.basename(norm_file).split(".")[0] == os.path.basename(f).split(".")[0]:
        #             import pdb
        #             pdb.set_trace()
        #         norm_data = np.load(norm_file)
        #         norm_scale,norm_offset = torch.from_numpy(norm_data["scale"]).float(),torch.from_numpy(norm_data["offset"]).float()
        #         pc = (pc + norm_offset) * norm_scale
        #
        #         lst.append(pc)

        self.point_clouds = lst
        # print("each pc shape: ", self.point_clouds[0].shape)
    def load_pc(self, i):
        f = self.pc_paths[i]
        pc = self.sample_pc(f, self.pc_size)

        norm_file = self.norm_files[i]
        if not os.path.basename(norm_file).split(".")[0] == os.path.basename(f).split(".")[0]:
            import pdb
            pdb.set_trace()
        norm_data = np.load(norm_file)
        norm_scale, norm_offset = torch.from_numpy(norm_data["scale"]).float(),torch.from_numpy(norm_data["offset"]).float()
        pc = (pc + norm_offset) * norm_scale

        return pc

    def get_all_files(self):
        return self.point_clouds, self.pc_paths

    def __getitem__(self, idx):
        if self.return_filename:
            return self.load_pc(idx), self.pc_paths[idx]
        else:
            return self.load_pc(idx)

    def __len__(self):
        return len(self.pc_paths)

    def sample_pc(self, f, samp=1024):
        '''
        f: path to csv file
        '''
        # data = torch.from_numpy(np.loadtxt(f, delimiter=',')).float()
        if os.path.basename(f).split(".")[-1] == "csv":
            data = torch.from_numpy(pd.read_csv(f, sep=',', header=None).values).float()
            pc = data[data[:, -1] == 0][:, :3]
        elif os.path.basename(f).split(".")[-1] == "ply":
            data = trimesh.load(f)
            pc = np.asarray(data.vertices)
            pc = torch.from_numpy(pc).float()
        else:
            raise NotImplementedError

        pc_idx = torch.randperm(pc.shape[0])[:samp]
        pc = pc[pc_idx]
        # print("pc shape, dtype: ", pc.shape, pc.dtype) # [1024,3], torch.float32
        # pc = normalize_pc(pc)
        # print("pc shape: ", pc.shape, pc.max(), pc.min())
        return pc

    def get_instance_filenames(self, data_source, split, filter_modulation_path=None, t="SdfSamples", suffix=".npz"):

        do_filter = filter_modulation_path is not None
        csvfiles = []
        for dataset in split:  # e.g. "acronym" "shapenet"
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    instance_filename = os.path.join(data_source, t, dataset, class_name, instance_name+suffix)

                    if do_filter:
                        mod_file = os.path.join(filter_modulation_path, class_name, instance_name, "latent.txt")

                        # do not load if the modulation does not exist; i.e. was not trained by diffusion model
                        if not os.path.isfile(mod_file):
                            continue

                    if not os.path.isfile(instance_filename):
                        logging.warning("Requested non-existent file '{}'".format(instance_filename))
                        raise Exception("Requested non-existent file '{}'".format(instance_filename))

                    csvfiles.append(instance_filename)
        return csvfiles