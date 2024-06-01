# %%
import trimesh
import numpy as np
import os
import json

import plotly.graph_objs as go
import open3d as o3d

from eval_iou import BoundingCubeNormalization, plotly_mesh3d_to_trimesh, trimesh_to_plotly_mesh3d

import json
import trimesh
import numpy as np
import os
import open3d as o3d

import plotly.graph_objs as go

from scipy.spatial import ConvexHull

import pdb

from tqdm import tqdm


def load_pcd_vis(pcd_path,pcd_name=None,sub_sample=50000) -> np.ndarray:
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
    
    N = point_clouds_data.shape[0]
    point_clouds_data = point_clouds_data[np.random.choice(N, sub_sample, replace=True), :]
    
    x = point_clouds_data[:, 0]
    y = point_clouds_data[:, 1]
    z = point_clouds_data[:, 2]
    trace = go.Scatter3d(x=x, y=y, z=z, 
            mode='markers',     
            marker=dict(
                size=0.5,  # Adjust the size of the markers here
                color='rgba(35, 35, 250, 0.8)'  # Set the color you want (e.g., light blue)
            ),
            name=pcd_name)
    return trace

def quaternion_list2rotmat(quant_list: list, format="xyzw"):
    assert len(quant_list) == 4, "Quaternion needs 4 elements"
    if format=="xyzw":
        q = np.quaternion(quant_list[0], quant_list[1], quant_list[2], quant_list[3])
    elif format=="wxyz":
        q = np.quaternion(quant_list[1], quant_list[2], quant_list[3], quant_list[0])
    R = quaternion.as_rotation_matrix(q)
    return R

def mesh_apply_rts(mesh, rotation_mat_c2w=np.eye(3), translation_c2w=np.zeros(3), scale_c2w=np.ones(3), mesh_name=None, mesh_color=None):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    transformed_vertices = (rotation_mat_c2w @ vertices.T).T - translation_c2w[np.newaxis, :]
    transformed_vertices = vertices * scale_c2w
    
    x, y, z = transformed_vertices.T  # Transposed for easier unpacking
    i, j, k = faces.T  # Unpack faces

    if mesh_color is None:
        mesh_color = "rgba(244,22,100,0.5)"

    mesh_transformed = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        color=mesh_color,
        name=mesh_name
    )
    return mesh_transformed

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

def mesh_load_go(mesh_path, scale_c2w=None, rotation_quat_wxyz=None, translation_c2w=None,mesh_name=None):
    # 从文件加载网格数据
    file_suffix = mesh_path.split(".")[-1]
    if file_suffix == "obj":
        with open(mesh_path, 'r') as file:
            lines = file.readlines()

        vertices = []
        faces = []

        for line in lines:
            if line.startswith('v '):
                vertex = line.split()[1:]
                vertices.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
            elif line.startswith('f '):
                face = line.split()[1:]
                face_indices = [int(idx.split('/')[0]) - 1 for idx in face]
                faces.append(face_indices)

        mesh = go.Mesh3d(x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
                        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces], name=mesh_name)
        return mesh

    elif file_suffix == "ply":
        from plyfile import PlyData

        # 从PLY文件加载网格数据
        plydata = PlyData.read(mesh_path)

        # 提取顶点坐标
        vertices = np.array([list(vertex) for vertex in plydata['vertex'].data])

        # 提取面数据
        faces = np.array(plydata['face'].data['vertex_indices'])
        faces = np.array([list(row) for row in faces])

        # 创建网格图形对象
        mesh = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=mesh_name)
        return mesh
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

def compute_pcr(points_gt, mesh_pred, threshold=0.047):
    # if sample_count:
    #     points_pred = mesh.sample(sample_count)
    # else:
    #     points_pred = mesh.vertices
    closest, distance, _ = trimesh.proximity.closest_point(mesh_pred, points_gt)
    pcr = (distance<threshold).mean()
    return pcr

# from evaluation_metrics import load_mesh, mesh_to_points, compute_convex_hull_volume, compute_iou

label2id = {
        "sofa": "04256520",
        "table": "04379243",
        "bed": "02818832",
        "bathtub": "02808440",
        "chair": "03001627", # 03001627
        "cabinet": "02933112",
        'plane':'02691156',
        'bottle':'02876657',
    }

# %%
label = "sofa"
gt_root = "/storage/user/huju/transferred/ws_dditnach/DATA/ScanARCW/canonical_mesh_manifoldplus"
gt_root = os.path.join(gt_root,label2id[label])

sdf_root = "/storage/user/huju/transferred/ws_dditnach/DeepSDF/data/SdfSamples/canonical_mesh_manifoldplus" # 加载相关的transformation

# output_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/49999/Meshes"
# output_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond/recon/23999/Meshes"
# output_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_neighbor/output/49999/test/mesh/Meshes"
output_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor/output/69999/test/mesh/Meshes"
pcd_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor/pcd"

mesh_list = os.listdir(output_root)
pcrs = []
base_names = []

for id_mesh, mesh in tqdm(enumerate(mesh_list)):

    # if id_mesh>9:
    #     break

    mesh_path1 = os.path.join(output_root,mesh)

    base_name = os.path.basename(mesh_path1).split(".")[0]
    base_names.append(base_name)

    mesh_id, scene1, scene2, _,  obj_id = base_name.split(".")[0].split("_")

    pcd_path = os.path.join(pcd_root, base_name+".pth.pcd")
    print(base_name)
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    trace = go.Scatter3d(
        x=points[:, 0],  # X 坐标
        y=points[:, 1],  # Y 坐标
        z=points[:, 2],  # Z 坐标
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8
        )
    )

    # pdb.set_trace()
    # 加载两个mesh文件
    # mesh1 = load_mesh(mesh_path1)
    mesh1 = load_mesh(mesh_path1)

    # mesh_path2 = os.path.join(gt_root,base_name, "model_normalized_manifoldplus.obj")
    mesh_path2 = os.path.join(gt_root,base_name,"model_canonical_manifoldplus.obj")
    mesh2 = load_mesh(mesh_path2)
    
    sdf_path = os.path.join(sdf_root,label2id[label],base_name+".npz")
    sdf = np.load(sdf_path)
    translation_mesh2sdf = sdf['translation_mesh2sdf']
    scale_mesh2sdf = sdf["scale_mesh2sdf"]

    mesh1 = mesh_apply_rts(mesh1,mesh_name="recon",mesh_color="rgba(22,244,244,0.5)")
    # mesh2 = mesh_apply_rts(mesh2,translation_c2w=translation_mesh2sdf,scale_c2w=scale_mesh2sdf,mesh_name="gt")
    mesh1_trimesh = plotly_mesh3d_to_trimesh(mesh1)
    mesh2, scale, translation = BoundingCubeNormalization(mesh2)
    # mesh2 = mesh_apply_rts(mesh2,mesh_name="gt")

    # input()

    print(mesh_path1)
    print(mesh_path2)

    vis_data = [mesh1,mesh2,trace]

    # closest, distance, _ = trimesh.proximity.closest_point(mesh1_trimesh, points)
    pcr = compute_pcr(points, mesh1_trimesh)
    print("pcr:{}".format(pcr))
    pcrs.append(pcr)

    layout = go.Layout(scene=dict(
            aspectmode='data',  # Set the aspect ratio to 'cube' for equal scales
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),)

    fig = go.Figure()
    fig = go.Figure(data=vis_data, layout=layout)
    # fig.show()

    # input()
    # break

# %%
pcrs = np.array(pcrs)
print("mean pcr:{}".format(pcrs.mean()))
pcr_path = output_root.replace("/Meshes","/pcrs.txt")
# np.savetxt(pcr_path,pcrs)

# mesh_list_path = output_root.replace("/Meshes","/mesh_pcr_list.txt")
with open(pcr_path,"w") as f:
    f.write("mean pcr:{}".format(pcrs.mean()))
    for i, imesh in enumerate(mesh_list):
        f.write(imesh+","+"{}".format(pcrs[i])+"\n")

