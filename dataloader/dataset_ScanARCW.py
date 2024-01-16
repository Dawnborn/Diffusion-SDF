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

import pdb
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

class MyScanARCWDataset(torch.utils.data.Dataset):
    def __init__(self,latent_path_root, pcd_path_root, json_file_root, sdf_file_root, split_file=None, pc_size=1024, sdf_size=20000):
        super().__init__()

        self.latent_path_root = latent_path_root  # /canonical_mesh_manifoldplus/04256520
        self.pcd_path_root = pcd_path_root  # DATA/ScanARCW/segmented_pcd
        self.json_file_root = json_file_root  # /DATA/ScanARCW/json_files_v5
        self.sdf_file_root = sdf_file_root  # /home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DATA/ScanARCW_new/ScanARCW/sdf_samples/04256520

        self.split_file = split_file
        self.pc_size = pc_size
        self.sdf_size = sdf_size

        self.conditional = bool(pcd_path_root)

        self.latent_paths = []
        self.pcd_paths = []

        self.allowed_suffix = ["txt","pth"]

        self.get_latent_and_pc_paths()

    def get_info_from_latent_name(self, l_name):
        # input should be like
        latent_name = l_name.split(".")[0]
        scene = latent_name.split("_")[1]+"_"+latent_name.split("_")[2]
        ins_id = latent_name.split("_")[-1]
        obj_id = latent_name.split("_")[0]
        return scene, ins_id, obj_id
    
    def get_latent_and_pc_paths(self):
        tmp_latent_paths = os.listdir(self.latent_path_root)

        tmp_pcd_paths = os.listdir(self.pcd_path_root)

        for i_name in tmp_latent_paths:
            if i_name.split(".")[-1] in self.allowed_suffix:
                self.latent_paths.append(os.path.join(self.latent_path_root, i_name))
                
    def load_latent(self, latent_path):
        suffix = latent_path.split(".")[-1]
        if suffix == "txt":
            latent = ( torch.from_numpy(np.loadtxt(latent_path)).float() )
        elif suffix == "pth":
            latent = (torch.load(latent_path))
        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError
        return latent

    def __len__(self):
        return len(self.latent_paths)

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

                print(instance_info.keys())
                pcd_world = load_pcd_raw(os.path.join(self.pcd_path_root, instance_info['segmented_cloud']))

                pcd_world = farthest_point_sample(pcd_world, npoint=self.pc_size)

                pcd_meshcoord = (
                            np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T
                pcd_sdfcoord = (pcd_meshcoord - translation_sdf2mesh[np.newaxis, :]) / scale_sdf2mesh

        return pcd_sdfcoord

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
        latent = self.load_latent(latent_path)
        latent_name = os.path.basename(latent_path).split(".")[0]
        gt_sdf = self.load_corresponding_sdf_path_of_latent(latent_name)

        if self.conditional:
            pc = self.load_corresponding_pcd_of_latent(latent_name)
            return {
                "latent" : latent,
                "gt_sdf_xyzv" : gt_sdf,
                "point_cloud": pc
            }
        else:
            return {
                "latent": latent,
                "gt_sdf_xyzv": gt_sdf
            }

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
                               pc_size=1024
                               )
    batch = dataset.check(1)
    import pdb
    pdb.set_trace()
