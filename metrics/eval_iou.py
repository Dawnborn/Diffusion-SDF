import trimesh
import numpy as np
from scipy.spatial import ConvexHull
import pdb
import os
from tqdm import tqdm

import plotly.graph_objs as go
import trimesh

# from evaluation_metrics import load_mesh, mesh_to_points, compute_convex_hull_volume, compute_iou

label2id = {
        "sofa": "04256520",
        # "table": "04379243",
        # "bed": "02818832",
        # "bathtub": "02808440",
        # "chair": "03001627",
        # "cabinet": "02933112",
        # 'plane':'02691156',
        # 'bottle':'02876657',
    }

def load_mesh(file_path):
    # 加载mesh文件
    return trimesh.load(file_path, force='mesh')

def mesh_to_points(mesh, count=100000):
    # 将mesh表面采样为点集
    return mesh.sample(count)

def compute_convex_hull_volume(points):
    # 计算点集的凸包体积
    hull = ConvexHull(points)
    return hull.volume

def compute_iou(mesh1, mesh2, sample_count=100000):
    # 将两个mesh表面采样为点集
    points1 = mesh_to_points(mesh1, count=sample_count)
    points2 = mesh_to_points(mesh2, count=sample_count)
    
    # 分别计算两个点集的凸包体积
    volume1 = compute_convex_hull_volume(points1)
    volume2 = compute_convex_hull_volume(points2)
    
    # 计算两个点集的并集体积
    union_points = np.concatenate([points1, points2], axis=0)
    union_volume = compute_convex_hull_volume(union_points)
    
    # 计算交集体积（使用容斥原理）
    intersection_volume = volume1 + volume2 - union_volume
    
    # 计算IoU
    iou = intersection_volume / union_volume
    return iou

def plotly_mesh3d_to_trimesh(plotly_mesh3d):
    """
    将Plotly Mesh3d对象转换为trimesh.Trimesh对象。
    
    参数:
        plotly_mesh3d (plotly.graph_objs.Mesh3d): Plotly Mesh3d对象。
        
    返回:
        trimesh.Trimesh: 转换后的trimesh对象。
    """
    # 从Plotly Mesh3d对象提取顶点坐标
    vertices = np.column_stack((plotly_mesh3d.x, plotly_mesh3d.y, plotly_mesh3d.z))
    
    # 从Plotly Mesh3d对象提取面
    faces = np.column_stack((plotly_mesh3d.i, plotly_mesh3d.j, plotly_mesh3d.k))
    
    # 创建trimesh.Trimesh对象
    trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return trimesh_obj

def BoundingCubeNormalization(mesh,buffer=1.03):
    """
    transferred from deepsdf preprocessing
    input: mesh: trimesh.load
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    xmax = vertices[:,0].max()
    ymax = vertices[:,1].max()
    zmax = vertices[:,2].max()

    xmin = vertices[:,0].min()
    ymin = vertices[:,1].min()
    zmin = vertices[:,2].min()

    xcenter = (xmax+xmin)/2.0
    ycenter = (ymax+ymin)/2.0
    zcenter = (zmax+zmin)/2.0

    vertices[:,0] -= xcenter
    vertices[:,1] -= ycenter 
    vertices[:,2] -= zcenter

    norms = np.linalg.norm(vertices, axis=1)

    max_dist = norms.max()
    max_dist *= buffer

    vertices /= max_dist

    x, y, z = vertices.T  # Transposed for easier unpacking
    i, j, k = faces.T  # Unpack faces

    # if mesh_color is None:
    mesh_color = "rgba(244,22,100,0.5)"
    mesh_name = "normalized"

    mesh_normalized = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        color=mesh_color,
        name=mesh_name
    )
    
    return mesh_normalized, 1./max_dist, np.array([xcenter,ycenter,zcenter])

if __name__ == '__main__':
    
    label = "sofa"
    gt_root = "/storage/user/huju/transferred/ws_dditnach/DATA/ScanARCW/canonical_mesh_manifoldplus"
    gt_root = os.path.join(gt_root,label2id[label])

    sdf_root = "/storage/user/huju/transferred/ws_dditnach/DeepSDF/data/SdfSamples/canonical_mesh_manifoldplus" # 加载相关的transformation

    output_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/49999/Meshes"
    # output_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond/recon/23999/Meshes"

    log_path = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/49999"
    # log_path = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond/recon/23999"
    
    log_path = os.path.join(log_path,"iou_log.txt")

    mesh_list = os.listdir(output_root)
    ious = []
    base_names = []

    for mesh in tqdm(mesh_list):
        mesh_path1 = os.path.join(output_root,mesh)

        base_name = os.path.basename(mesh_path1).split(".")[0]
        base_names.append(base_name)

        mesh_id, scene1, scene2, _,  obj_id = base_name.split(".")[0].split("_")
        
        # pdb.set_trace()
        # 加载两个mesh文件
        mesh1 = load_mesh(mesh_path1)

        # mesh_path2 = os.path.join(gt_root,base_name, "model_normalized_manifoldplus.obj")
        mesh_path2 = os.path.join(gt_root,base_name,"model_canonical_manifoldplus.obj")
        mesh2 = load_mesh(mesh_path2)
        mesh2, scale, translation = BoundingCubeNormalization(mesh2)
        mesh2 = plotly_mesh3d_to_trimesh(mesh2)

        # 计算IoU
        iou = compute_iou(mesh1, mesh2)
        print(f"IoU: {iou}")
        ious.append(iou)

    with open(log_path,"w") as f:
        ious = np.array(ious)
        f.write("{}:{}\n".format("mean",ious.mean()))
        for base_name, iou in zip(base_names,ious):
            f.write("{}, {}\n".format(base_name,iou))
    print("results saved to: {}".format(log_path))
        

        
