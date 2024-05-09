import torch
import sys
import os, json
from pathlib import Path
from scipy.spatial.transform import Rotation as Rotation
import datetime
import numpy as np
import open3d as o3d
from typing import Iterator, List, Tuple, Dict
import random
import shutil
import matplotlib.cm as cm
import time
from torch.utils.data import Sampler, BatchSampler
from torch.utils.data.sampler import Sampler

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import torch.utils.data as Data

sys.path.append(str(Path(__file__).resolve().parent.parent))

import random

import pdb
import pandas as pd

from tqdm import tqdm
# class Instance:
#     def __init__(self, instance_tuple: Tuple, data_cfg):
#         pass

# 'scene_name': instance_index_["scene_name"], 'id': instance_index_["id"],
#                 'category_name': category_name,
def boundary2categ(categ_ind_boundary:dict, index):
    for categ, boundary in categ_ind_boundary.items():
        if index>=boundary[0] and index<boundary[1]:
            return categ
    return None

def quaternion_list2rotmat(quant_list: list):
    assert len(quant_list) == 4, "Quaternion needs 4 elements"
    # q = np.quaternion(quant_list[0], quant_list[1], quant_list[2], quant_list[3]) # np.quaternion(w,x,y,z), wbh used json version is actually wxyz, but marked as xyzw
    # R = quaternion.as_rotation_matrix(q)
    q = Rotation.from_quat([quant_list[1], quant_list[2], quant_list[3],quant_list[0]])
    R = q.as_matrix()
    return R

def random_point_sample(point: np.ndarray, npoint: int):
    np.random.seed(int(datetime.datetime.now().timestamp())%1000)
    N, D = point.shape
    if N > npoint:
        point = point[np.random.choice(np.arange(N), npoint, replace = False)]
    else:
        point = np.concatenate([point,
                                point[np.random.choice(np.arange(N), npoint-N, replace = True)]],
                                axis = 0
        )
    return point

def farthest_point_sample(point: np.ndarray, npoint: int):
    N, D = point.shape
    if N < npoint:
        point = np.concatenate([point,
                                point[np.random.choice(np.arange(N), npoint - N, replace=True)]],
                               axis=0
                               )
        return point

    if N > 2 * npoint:
        point = random_point_sample(point, 2 * npoint)
        N, D = point.shape

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def load_pcd_raw(pcd_path) -> np.ndarray:
    _, file_extension = os.path.splitext(pcd_path)
    if file_extension == ".pcd":
        pcd = o3d.io.read_point_cloud(pcd_path)
        point_clouds_data = np.asarray(pcd.points)
    elif file_extension == ".txt":
        point_clouds_data = np.loadtxt(pcd_path)
    elif file_extension == ".npy":
        point_clouds_data = np.load(pcd_path)
    else:
        raise NotImplementedError

    return point_clouds_data

def remove_nan(tmp):
    mask = np.isnan(tmp[:,3])
    return tmp[~mask]

num2cat = {
    "02808440": "bathtub",
    "02818832": "bed",
    "02871439": "bookshelf",
    "02933112": "cabinet",
    "03001627": "chair",
    "04256520": "sofa",
    "04379243": "table"
}

cat2num = dict()
for key in num2cat.keys():
    cat2num[num2cat[key]] = key

class MyScanARCWDataset(torch.utils.data.Dataset):
    def __init__(self,latent_path_root, pcd_path_root, json_file_root, sdf_file_root, split_file=None, pc_size=1024, sdf_size=20000, conditional=True, use_sdf=False, length=-1, times=1, pre_load=False, include_category=False, preprocess=None, use_neighbor=False, mode="train",specs=None):
        super().__init__()

        self.specs = specs

        self.latent_path_root = latent_path_root  # /canonical_mesh_manifoldplus/04256520
        self.pcd_path_root = pcd_path_root  # DATA/ScanARCW/segmented_pcd
        self.json_file_root = json_file_root  # /DATA/ScanARCW/json_files_v5
        self.sdf_file_root = sdf_file_root  # /home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520
        
        self.split_file = split_file
        self.preprocess = preprocess
        self.mode = mode

        self.pc_size = pc_size
        self.sdf_size = sdf_size

        self.conditional = (conditional and bool(pcd_path_root))
        self.use_neighbor = use_neighbor and bool(preprocess)

        self.pre_load = pre_load
        self.use_sdf = use_sdf

        self.latent_paths = []
        self.latent_dict = {}

        self.pcd_paths = []
        self.pcd_dict = {}
        self.neighbor_pcd_dict = {}

        # self.mesh_paths = []
        self.mesh_paths_dict = {}
        self.transformations_dict = {}

        self.allowed_suffix = ["txt","pth"]

        self.length = length

        self.times = times

        self.get_latent_and_pc_paths()

    def save_latent_paths(self, path="output.txt"):
        with open(path, "w") as file:
            for item in self.latent_paths:
                file.write("%s\n" % item)
        print("list is saved to {}".format(path))

    def get_info_from_latent_name(self, l_name):
        # input should be like: 1a4a8592046253ab5ff61a3a2a0e2484_scene0484_00_ins_1.pth
        l_name = os.path.basename(l_name)
        latent_name = l_name.split(".")[0]
        scene = latent_name.split("_")[1]+"_"+latent_name.split("_")[2]
        ins_id = latent_name.split("_")[-1]
        obj_id = latent_name.split("_")[0]
        return scene, ins_id, obj_id

    def pcd_dict_interst(self, scene_name,ins_id, obj):
        if scene_name in self.pcd_dict.keys():
            self.pcd_dict[scene_name][ins_id] = obj
        else:
            self.pcd_dict[scene_name] = dict()
            self.pcd_dict[scene_name][ins_id] = obj

    def get_latent_and_pc_paths(self):
        """
                initialization of the class
        :return:
        """
        tmp_latent_paths = sorted(os.listdir(self.latent_path_root))
        print("======in all {}".format(len(tmp_latent_paths)))

        # if split_file, only support single category!
        if self.preprocess:
            print("=========start indexing latents from preprocessing!")

            # # 有错误，有些场景里的latent没有用到在train_set里
            # train_split = os.path.join(self.preprocess, "scene_split.json")
            # with open(train_split, 'r') as file:
            #     data = json.load(file)
            # data[self.mode]
            # new_tmp_latent_paths = []
            # for latent in tmp_latent_paths:
            #     scene, ins_id, obj_id = self.get_info_from_latent_name(latent)
            #     if scene in data[self.mode]:
            #         new_tmp_latent_paths.append(latent)
            # tmp_latent_paths = new_tmp_latent_paths

            json_file_path = os.path.join(self.preprocess,"{}_set.json".format(self.mode))
            with open(json_file_path, "r") as f:
                data = json.load(f)
            rows = []
            # 遍历JSON中的每个对象
            for cat in data["data"]:
                for item in tqdm(data["data"][cat]):
                    scene_name = item['scene_name']
                    obj_id = item['id']
                    category = item['category']
                    neighbour_ids = item['neighbour_id']
                    neighbour_ious = item['neighbour_iou']
                    row = {
                        'scene_name': scene_name,
                        'id': obj_id,
                        'category': category,
                        'neighbour_ids': neighbour_ids,
                        'neighbour_ious': neighbour_ious,
                    }
                    rows.append(row)
            # 使用rows列表创建DataFrame
            df = pd.DataFrame(rows)
            df.set_index(['scene_name', 'id'], inplace=True)
            self.neighbor_info_df = df
            
            new_tmp_latent_paths = []
            for latent in tmp_latent_paths:
                scene_name, ins_id, obj_id = self.get_info_from_latent_name(l_name=latent)
                if (scene_name, ins_id) in self.neighbor_info_df.index:
                    new_tmp_latent_paths.append(latent)
            tmp_latent_paths = new_tmp_latent_paths
            print("=====using {} for {}".format(len(tmp_latent_paths), self.mode))

        current_length = 0
        if self.length == -1:
            self.length = len(tmp_latent_paths)

        print("checking corresponding segmented pcd...")
        for i_name in tqdm(tmp_latent_paths):
            if not i_name.split(".")[-1] in self.allowed_suffix:
                continue
            pcd_path = self.check_corresponding_pcd_of_latent(i_name)
            if not pcd_path:
                continue
            i_latent_path = os.path.join(self.latent_path_root, i_name)
            self.latent_paths.append(i_latent_path)

            scene_name, ins_id, obj_id = self.get_info_from_latent_name(l_name=latent)
            if self.pre_load:
                self.latent_dict[i_latent_path] = self.load_latent(i_latent_path)
                if self.conditional:
                    self.pcd_dict[i_name.split(".")[0]] = self.load_corresponding_pcd_of_latent(i_name)
                    # pcd = self.load_corresponding_pcd_of_scene_id_preloaded(scene_name, ins_id)
            current_length += 1
            if current_length >= self.length:
                break

        if self.use_neighbor:
            if self.pre_load:
                # 预加载neighbor pcds
                print("preloading neighbor pcds...")
                for i_lat in tqdm(self.latent_dict.keys()):
                    neighbor_pcds = self.load_corresponding_neighbor_pcd_of_latent(i_lat)
                    self.neighbor_pcd_dict[i_lat] = neighbor_pcds
            else:
                pass
        
        self.latent_paths = self.latent_paths*self.times
        print("=========latent indexing finished!")

    def load_latent_preloaded(self, latent_path):
        if self.pre_load and latent_path in self.latent_dict.keys():
            return self.latent_dict[latent_path]
        else:
            raise RuntimeError

    def load_latent(self, latent_path):
        suffix = latent_path.split(".")[-1]
        if suffix == "txt":
            latent = ( torch.from_numpy(np.loadtxt(latent_path)).float().cpu().numpy())
        elif suffix == "pth":
            latent = (torch.load(latent_path).cpu().numpy())
        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError
        return latent

    def __len__(self):
        return len(self.latent_paths)

    def check_corresponding_pcd_of_latent(self, latent_name):
        scene_name, ins_id, obj_id = self.get_info_from_latent_name(l_name=latent_name)
        json_path = os.path.join(self.json_file_root,scene_name+".json")
        if not os.path.isfile(json_path):
            import pdb
            pdb.set_trace()
            raise RuntimeError
        else:
            with open(json_path,'r') as f: # '/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/output/default_experiment_3_class/test_set.json'
                raw = json.load(f)
                instances_dict = raw[scene_name]['instances']
                instance_info = instances_dict[ins_id]

                # print(instance_info.keys())
                pcd_file = instance_info.get('segmented_cloud', None)
                if not pcd_file:
                    print(scene_name)
                    print(ins_id)
                    print(instance_info)
                    return False
                    
                pcd_path = os.path.join(self.pcd_path_root, pcd_file)
                if not os.path.isfile(pcd_path):
                    print(instance_info)
                    return False
                
                scale_sdf2mesh = np.array(instance_info["scale_sdf2mesh"])  # 1,
                translation_sdf2mesh = np.array(instance_info["translation_sdf2mesh"])  # 3,
                gt_translation_c2w = np.array(instance_info["gt_translation_c2w"])  # 3,
                gt_rotation_mat_c2w = quaternion_list2rotmat(instance_info["gt_rotation_quat_wxyz_c2w"])

                transformation_c2w = {}
                transformation_c2w["scale_sdf2mesh"] = scale_sdf2mesh
                transformation_c2w["translation_sdf2mesh"] = translation_sdf2mesh
                transformation_c2w["gt_translation_c2w"] = gt_translation_c2w
                transformation_c2w["gt_rotation_quat_wxyz_c2w"] = gt_rotation_mat_c2w

                self.transformations_dict[latent_name] = transformation_c2w
                self.mesh_paths_dict[latent_name] = instance_info["gt_scaled_canonical_mesh"]

        return pcd_path

    def load_corresponding_pcd_of_latent(self, latent_name):
        scene_name, ins_id, obj_id = self.get_info_from_latent_name(l_name=latent_name)
        json_path = os.path.join(self.json_file_root,scene_name+".json")
        if not os.path.isfile(json_path):
            import pdb
            pdb.set_trace()
            raise RuntimeError
        else:
            with open(json_path,'r') as f: # '/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/output/default_experiment_3_class/test_set.json'
                raw = json.load(f)
                instances_dict = raw[scene_name]['instances']
                instance_info = instances_dict[ins_id]

                scale_sdf2mesh = np.array(instance_info["scale_sdf2mesh"])  # 1,
                translation_sdf2mesh = np.array(instance_info["translation_sdf2mesh"])  # 3,
                gt_translation_c2w = np.array(instance_info["gt_translation_c2w"])  # 3,
                gt_rotation_mat_c2w = quaternion_list2rotmat(instance_info["gt_rotation_quat_wxyz_c2w"])

                # print(instance_info.keys())
                pcd_file = instance_info.get('segmented_cloud', None)
                if not pcd_file:
                    print(scene_name)
                    print(ins_id)
                    print(instance_info)
                    import pdb; pdb.set_trace()
                pcd_path = os.path.join(self.pcd_path_root, pcd_file)
                if not os.path.isfile(pcd_path):
                    print(instance_info)
                    import pdb; pdb.set_trace()
                pcd_world = load_pcd_raw(pcd_path)

                pcd_world = farthest_point_sample(pcd_world, npoint=self.pc_size)

                pcd_meshcoord = (
                            np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T
                pcd_sdfcoord = (pcd_meshcoord - translation_sdf2mesh[np.newaxis, :]) / scale_sdf2mesh

        return pcd_sdfcoord.astype(np.float32)

    def load_corresponding_pcd_of_scene_id(self, scene_name, ins_id):
        # scene_name, ins_id, obj_id = self.get_info_from_latent_name(l_name=latent_name)
        json_path = os.path.join(self.json_file_root,scene_name+".json")
        if not os.path.isfile(json_path):
            import pdb
            pdb.set_trace()
            raise RuntimeError
        else:
            with open(json_path,'r') as f: # '/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/output/default_experiment_3_class/test_set.json'
                raw = json.load(f)
                instances_dict = raw[scene_name]['instances']
                instance_info = instances_dict[ins_id]

                scale_sdf2mesh = np.array(instance_info["scale_sdf2mesh"])  # 1,
                translation_sdf2mesh = np.array(instance_info["translation_sdf2mesh"])  # 3,
                gt_translation_c2w = np.array(instance_info["gt_translation_c2w"])  # 3,
                gt_rotation_mat_c2w = quaternion_list2rotmat(instance_info["gt_rotation_quat_wxyz_c2w"])

                # print(instance_info.keys())
                pcd_file = instance_info.get('segmented_cloud', None)
                if not pcd_file:
                    print(scene_name)
                    print(ins_id)
                    print(instance_info)
                    import pdb; pdb.set_trace()
                pcd_path = os.path.join(self.pcd_path_root, pcd_file)
                if not os.path.isfile(pcd_path):
                    print(instance_info)
                    import pdb; pdb.set_trace()
                pcd_world = load_pcd_raw(pcd_path)

                pcd_world = farthest_point_sample(pcd_world, npoint=self.pc_size)

                pcd_meshcoord = (
                            np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T
                pcd_sdfcoord = (pcd_meshcoord - translation_sdf2mesh[np.newaxis, :]) / scale_sdf2mesh

        return pcd_sdfcoord.astype(np.float32)

    def load_corresponding_pcd_of_scene_id_preloaded(self,scene_name,ins_id):
        try:
            pcd = self.pcd_dict[scene_name][ins_id]
        except:
            print("caching scene {} ins {}".format(scene_name, ins_id))
            pcd = self.load_corresponding_pcd_of_scene_id(scene_name, ins_id)
            self.pcd_dict_interst(scene_name, ins_id, pcd)
        return pcd


    def load_corresponding_pcd_of_latent_preloaded(self,latent_name):
        if self.pre_load and latent_name in self.pcd_dict.keys():
            return self.pcd_dict[latent_name]
        else:
            print("latent {} not pre_loaded!!!".format(latent_name))
            import pdb; pdb.set_trace()
            raise RuntimeError

    def load_corresponding_neighbor_pcd_of_latent(self, i_name):
        """
        return neighbor pcds of shape [num_neighbors, num_points, 3], padding with self point cloud if not enough neighbors
        """
        scene_name, ins_id, obj_id = self.get_info_from_latent_name(i_name)

        neighbor_info = self.neighbor_info_df.loc[(scene_name, ins_id)]
        neighbor_pcds = []

        for id, ng in enumerate(neighbor_info['neighbour_ids']):
            if id>=self.specs["max_neighbor"]:
                break
            neighbor_pcd = self.load_corresponding_pcd_of_scene_id(scene_name, ng)
            # if self.pre_load:
            #     neighbor_pcd = self.load_corresponding_pcd_of_scene_id_preloaded(scene_name, ng)
            # else:
            #     neighbor_pcd = self.load_corresponding_pcd_of_scene_id(scene_name, ng)
            neighbor_pcds.append(neighbor_pcd)

        inner_pcd = self.load_corresponding_pcd_of_scene_id(scene_name, ins_id)
        # if self.pre_load:
        #     inner_pcd = self.load_corresponding_pcd_of_scene_id_preloaded(scene_name, ins_id)
        # else:
        #     inner_pcd = self.load_corresponding_pcd_of_scene_id(scene_name, ins_id)
        while(len(neighbor_pcds)<self.specs["max_neighbor"]):
            neighbor_pcds.append(inner_pcd)

        assert(len(neighbor_pcds)==self.specs["max_neighbor"])

        neighbor_pcds = np.stack(neighbor_pcds,axis=0)
        return neighbor_pcds

    def load_corresponding_neighbor_pcd_of_latent_preloaded(self, latent_path):
        "i_path: full path of latent"
        if self.pre_load and latent_path in self.neighbor_pcd_dict.keys():
            return self.neighbor_pcd_dict[latent_path]
        else:
            print("neighbor pcds of latent {} not pre_loaded!!!".format(latent_path))
            print("caching neighbor pcds of latent {}!!!".format(latent_path))
            # neighbor_pcds = self.load_corresponding_neighbor_pcd_of_latent(latent_path)
            # self.neighbor_pcd_dict[latent_path] = neighbor_pcds
            # return neighbor_pcds
            raise RuntimeError

    def get_corresponding_sdf_path_of_latent(self,latent_name):
        sdf_name = latent_name.split(".")[0] + ".npz"
        sdf_path = os.path.join(self.sdf_file_root,sdf_name)

        return sdf_path

    def load_corresponding_sdf_path_of_latent(self,latent_name):
        # import pdb
        # pdb.set_trace()
        sdf_path = self.get_corresponding_sdf_path_of_latent(latent_name)
        gt_sdf_xyzv = np.load(sdf_path)

        gt_sdf_xyzv_pos = remove_nan(gt_sdf_xyzv['pos'])
        gt_sdf_xyzv_neg = remove_nan(gt_sdf_xyzv['neg'])
        half_neg = True
        if half_neg:
            gt_sdf_xyzv_pos = random_point_sample(gt_sdf_xyzv_pos, npoint=int(self.sdf_size/2))
            gt_sdf_xyzv_neg = random_point_sample(gt_sdf_xyzv_neg, npoint=int(self.sdf_size/2))
            gt_sdf_xyzv = np.concatenate([gt_sdf_xyzv_pos, gt_sdf_xyzv_neg], axis=0)
        else:
            gt_sdf_xyzv = random_point_sample(gt_sdf_xyzv_pos, npoint=int(self.sdf_size))

        return gt_sdf_xyzv

    def __getitem__(self,index):
        latent_path = self.latent_paths[index]
        latent_name = os.path.basename(latent_path).split(".")[0]
        scene_name, ins_id, obj_id = self.get_info_from_latent_name(latent_path)
        ans_dict = {}

        if self.pre_load:
            latent = self.load_latent_preloaded(latent_path)
        else:
            latent = self.load_latent(latent_path)
        ans_dict["latent"] = latent
        ans_dict["latent_path"] = latent_path

        if self.use_sdf:
            gt_sdf = self.load_corresponding_sdf_path_of_latent(latent_name)
            ans_dict['gt_sdf_xyzv'] = gt_sdf

        # if self.ret_mesh_path:
        #     mesh_path = self.load_corresponding_mesh_path_of_latent(latent_name)
        #     ans_dict['mesh_path'] = mesh_path
        #     ans_dict['mesh_scale'] = 

        if self.conditional:
            if self.pre_load:
                pc = self.load_corresponding_pcd_of_latent_preloaded(latent_name)
                # pc = self.load_corresponding_pcd_of_scene_id_preloaded(scene_name, ins_id)
            else:
                pc = self.load_corresponding_pcd_of_latent(latent_name)
                # pc = self.load_corresponding_pcd_of_scene_id(scene_name, ins_id)
            ans_dict['point_cloud'] = pc

        if self.use_neighbor:
            if not self.pre_load:
                neighbor_pcds = self.load_corresponding_neighbor_pcd_of_latent(latent_name)
            else:
                neighbor_pcds = self.load_corresponding_neighbor_pcd_of_latent_preloaded(latent_path)
            ans_dict['neighbor_pcds'] = neighbor_pcds

        return ans_dict

    def check(self, index, output_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/output"):
        batch_ = self.__getitem__(index)
        #! point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch_["point_cloud"])
        output_path = f"{output_root}/check_pcd_canonical_{index}.ply"
        o3d.io.write_point_cloud(output_path, pcd)
        print("test point cloud is written to {}".format(output_path))
        # #! mesh
        # mesh_ = o3d.io.read_triangle_mesh(os.path.join(self.dataset_rootdir_, batch_["mesh_path"]))
        # scale_sdf2mesh = np.array(batch_["scale_sdf2mesh"]) # 1,
        # translation_sdf2mesh = np.array(batch_["translation_sdf2mesh"]) # 3,
        # mesh_ = mesh_.translate(-translation_sdf2mesh)
        # mesh_.scale(1/scale_sdf2mesh, center=(0, 0, 0))
        # o3d.io.write_triangle_mesh(f'output/data_check/check_mesh_canonical_{index}.obj', mesh_)

        # shutil.copy(os.path.join(self.dataset_rootdir_, batch_["mesh_path"]),
        #              "check_mesh_canonical.obj")
        #! sdf
        xyz = batch_["gt_sdf_xyzv"][:, :3]
        color_values = np.abs(batch_["gt_sdf_xyzv"][:, 3])
        colors = cm.get_cmap('Greys_r')(color_values)[:, :3] # 取RGB，忽略透明度

        pcd_sdf = o3d.geometry.PointCloud()
        pcd_sdf.points = o3d.utility.Vector3dVector(xyz)
        pcd_sdf.colors = o3d.utility.Vector3dVector(colors)
        output_path_sdf = f"{output_root}/check_sdf_canonical_{index}.ply"
        o3d.io.write_point_cloud(output_path_sdf, pcd_sdf)
        print("test sdf is written to {}".format(output_path_sdf))

        return batch_

if __name__ == "__main__":
    dataset = MyScanARCWDataset(latent_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24/LatentCodes/train/2000/canonical_mesh_manifoldplus/04256520",
                               pcd_path_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA",
                               json_file_root="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/json_files_v5",
                               sdf_file_root="/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520",
                               pc_size=6000,
                               length=10, # debug length
                               times=1,
                               pre_load=True,
                               conditional=True,
                               include_category=False,
                               use_neighbor=True,
                                use_sdf=True,
                               preprocess="/storage/user/huju/transferred/ws_dditnach/DDIT/preprocess_output/experiment_1_class_alter_NptcUsdf_repro"
                               )
    l = len(dataset)
    for i in range(10):
        id = random.randint(0,l-1)
        batch = dataset.check(id)
        # except:
        #     import pdb
        #     pdb.set_trace()
