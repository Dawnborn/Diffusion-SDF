import json
import trimesh
import numpy as np
np.random.seed(42)
import quaternion
import os
import open3d as o3d

import pandas as pd 
import plotly.graph_objs as go
import plotly.io as pio

cat2color_map = {
    'sofa': 'rgba(244, 22, 100, 1)',    # 粉红色
    'table': 'rgba(22, 244, 100, 1)',   # 绿色
    'chair': 'rgba(100, 22, 244, 1)',   # 紫色
    'bed': 'rgba(244, 100, 22, 1)',      # 橙色
    'bookshelf': 'rgba(22, 100, 244, 1)',  # 蓝色
    'cabinet': 'rgba(100, 244, 22, 1)'     # 青色
}

pcd_color = "rgba(10, 10, 10, 0.5)"

def get_info_from_latent_name(l_name):
    # input should be like: 1a4a8592046253ab5ff61a3a2a0e2484_scene0484_00_ins_1.pth
    l_name = os.path.basename(l_name)
    latent_name = l_name.split(".")[0]
    scene = latent_name.split("_")[1]+"_"+latent_name.split("_")[2]
    ins_id = latent_name.split("_")[-1]
    obj_id = latent_name.split("_")[0]
    return scene, ins_id, obj_id

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
                size=0.8,  # Adjust the size of the markers here
                color=pcd_color  # Set the color you want (e.g., light blue)
            ),
            name=pcd_name)
    return trace

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

def BoundingCubeDeNormalization(mesh, max_dists_d, centers, buffer=1.03):
    """
    transferred from deepsdf preprocessing
    input: mesh: trimesh.load
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    max_dist = 1./max_dists_d

    max_dist /= buffer

    vertices *= max_dist

    vertices[:,0] += centers[0]
    vertices[:,1] += centers[1]
    vertices[:,2] += centers[2]

    x, y, z = vertices.T  # Transposed for easier unpacking
    i, j, k = faces.T  # Unpack faces

    # if mesh_color is None:
    mesh_color = "rgba(244,22,100,0.5)"
    mesh_name = "unnormalized"

    mesh_unnormalized = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        color=mesh_color,
        name=mesh_name
    )
    
    return mesh_unnormalized

def quaternion_list2rotmat(quant_list: list, format="xyzw"):
    assert len(quant_list) == 4, "Quaternion needs 4 elements"
    if format=="xyzw":
        q = np.quaternion(quant_list[0], quant_list[1], quant_list[2], quant_list[3])
    elif format=="wxyz":
        q = np.quaternion(quant_list[1], quant_list[2], quant_list[3], quant_list[0])
    R = quaternion.as_rotation_matrix(q)
    return R

def mesh_apply_rts(mesh, rotation_mat_c2w=None, translation_c2w=None, scale_c2w=None, mesh_name=None, mesh_color=None):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    
    if rotation_mat_c2w is None:
    # Apply rotation
    # pcd_meshcoord = (np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T
        transformed_vertices = vertices
    else:
        transformed_vertices = (rotation_mat_c2w @ vertices.T).T + translation_c2w[np.newaxis, :]
    
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

def mesh_load(mesh_path, scale_c2w=None, rotation_quat_wxyz=None, translation_c2w=None,mesh_name=None):
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

test_pred_path = {
    "sofa":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor/output/69999/test/mesh/Meshes",
    "table":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_table_train_noneighbor/output/50999/test/mesh/Meshes",
    # "chair":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddi"t_stage2_diff_cond_chair_train_neighbor/output/18999/test/mesh/Meshes", #TO BE CORRECT
    "chair":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_noneighbor/output/18999/test/mesh/oldMeshes",
    "bed":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_bed_train_noneighbor/output/55999/test/mesh",
    "bookshelf":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_bookshelf_train_noneighbor/output/23999/test/mesh",
    "cabinet":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_cabinet_train_noneighbor/output/23999/test/mesh"
}

val_pred_path = {
    "sofa":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor/output/69999/val/mesh/Meshes",
    "table":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_table_train_noneighbor/output/50999/val/mesh",
    # "chair":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_neighbor/output/18999/val/mesh/Meshes", #TO BE CORRECT
    "chair":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_noneighbor/output/18999/val/mesh/Meshes",
    "bed":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_bed_train_noneighbor/output/55999/val/mesh",
    "bookshelf":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_bookshelf_train_noneighbor/output/23999/val/mesh",
    "cabinet":"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_cabinet_train_noneighbor/output/23999/val/mesh"
}

all_list_diff = {}
category_list = {}
# statis_dict = pd.DataFrame(columns=list(test_pred_path.keys()))
statis_dict = pd.DataFrame(columns=["scene"]+list(test_pred_path.keys()))
statis_dict.set_index("scene", inplace=True)

for key,val in test_pred_path.items():
    for mesh in os.listdir(val):
        basename = mesh.split(".")[0]
        all_list_diff[basename] = (key,os.path.join(val,mesh))

        scene_name, ins_id, obj_id = get_info_from_latent_name(basename)

        try:
            statis_dict.loc[scene_name, key] += 1
        except:
            new_row = {"scene":scene_name, "sofa":0, "table":0, "chair":0, "bed":0, "bookshelf":0, "cabinet":0}
            statis_dict.loc[scene_name] = new_row
            statis_dict.loc[scene_name, key] += 1

for key,val in val_pred_path.items():
    for mesh in os.listdir(val):
        basename = mesh.split(".")[0]
        all_list_diff[basename] = (key,os.path.join(val,mesh))

        scene_name, ins_id, obj_id = get_info_from_latent_name(basename)

        try:
            statis_dict.loc[scene_name, key] += 1
        except:
            new_row = {"scene":scene_name, "sofa":0, "table":0, "chair":0, "bed":0, "bookshelf":0, "cabinet":0}
            statis_dict.loc[scene_name] = new_row
            statis_dict.loc[scene_name, key] += 1


from tqdm import tqdm

for scene_name in tqdm(list(statis_dict.index)):
    data_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/DATA"
    json_version = "json_files_v5"
    traj_file = None

    # LAYOUT_VIS=True
    LAYOUT_VIS=True

    # MESH_VIS=True
    # MESH_PRED_VIS=False
    MESH_VIS=False
    MESH_PRED_VIS=True

    # SEGPCD_VIS=True
    # COMPLETE_PCD_VIS=False
    SEGPCD_VIS=True
    COMPLETE_PCD_VIS=False

    M = 50000

    vis_data = []
    mesh_paths = []

    if COMPLETE_PCD_VIS:
        pcd_path = os.path.join(data_root,"ScanARCW/complete_pcd/{}/{}.txt".format(scene_name,scene_name))
        data = np.loadtxt(pcd_path)
        N = data.shape[0]
        data = data[np.random.choice(N, M, replace=False), :]
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        complete_pcd = go.Scatter3d(x=x, y=y, z=z, 
                mode='markers',     
                marker=dict(
                    size=0.8,  # Adjust the size of the markers here
                    color=pcd_color  # Set the color you want (e.g., light blue)
                ),
                name="scene points")
        vis_data.append(complete_pcd)
            

    # json_file = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/DATA/ScanARCW/json_files_v2/{}.json".format(scene_name)
    json_file = os.path.join(data_root,"ScanARCW",json_version,"{}.json".format(scene_name))

    with open(json_file, 'r') as file:
        raw_json = json.load(file)
    for scene_id in raw_json.keys():
        instances_data = raw_json[scene_id]["instances"]
        for instance_id in instances_data.keys():
            instance_data = instances_data[instance_id]
            if not instance_data:
                print("The dictionary is empty.")
                continue
            
            if (instance_data['category_name'] in ["window","door"]):
                if SEGPCD_VIS:
                    seg_pcd_path = os.path.join(data_root,instance_data["segmented_cloud"])
                    seg_pcd = load_pcd_vis( seg_pcd_path, sub_sample=int( M/len(instances_data.keys()) ) )
                    seg_pcd.name = "pcd_{}".format(instance_id)
                    # seg_pcd['color'] = pcd_color
                    vis_data.append(seg_pcd)


            if (instance_data['category_name'] in ["layout"]):
                if SEGPCD_VIS:
                    seg_pcd_path = os.path.join(data_root,instance_data["segmented_cloud"])
                    seg_pcd = load_pcd_vis( seg_pcd_path )
                    seg_pcd.name = "pcd_{}".format(instance_id)
                    # seg_pcd['color'] = pcd_color
                    vis_data.append(seg_pcd)

            gt_translation_c2w = np.array(instance_data["gt_translation_c2w"]) # 3,
            mesh_path = os.path.join(data_root,instance_data["gt_scaled_canonical_mesh"])
            if instance_data.get("gt_rotation_quat_xyzw_c2w",False):
                print(mesh_path)
                gt_rotation_mat_c2w = quaternion_list2rotmat(instance_data["gt_rotation_quat_xyzw_c2w"],format="xyzw")
                if instance_data['category_id'] in ["1","2","3","4"]:
                    print(instance_data['category_id'])
                    gt_rotation_mat_c2w = quaternion_list2rotmat(instance_data["gt_rotation_quat_xyzw_c2w"],format="wxyz")
            elif instance_data.get("gt_rotation_quat_wxyz_c2w",False):
                print(mesh_path)
                gt_rotation_mat_c2w = quaternion_list2rotmat(instance_data["gt_rotation_quat_wxyz_c2w"])
            else:
                print("skipped!")
                continue
            
            # mesh = mesh_load(mesh_path)
            mesh_paths.append(mesh_path)
            mesh = trimesh.load(mesh_path)
            basename = os.path.basename(os.path.dirname(mesh_path))
            mesh_normalied, scale, centers = BoundingCubeNormalization(mesh,buffer=1.03)
            mesh = mesh_apply_rts(mesh, mesh_name=basename, rotation_mat_c2w=gt_rotation_mat_c2w, translation_c2w=gt_translation_c2w, scale_c2w=None)
            
            # 加载gt mesh计算normal参数然后反向加载到重建结果上再加变换回场景
            if basename in all_list_diff.keys():
                cat, mesh_pred_path = all_list_diff[basename]
                mesh_pred = trimesh.load(mesh_pred_path)
                mesh_pred_denormalized = BoundingCubeDeNormalization(mesh_pred, scale, centers)
                mesh_pred_denormalized = plotly_mesh3d_to_trimesh(mesh_pred_denormalized)
                mesh_pred_denormalized = mesh_apply_rts(mesh_pred_denormalized, mesh_name="pred_"+basename, rotation_mat_c2w=gt_rotation_mat_c2w, translation_c2w=gt_translation_c2w, scale_c2w=None)
                
                if MESH_VIS:
                    mesh['color'] = cat2color_map[cat]
                    mesh['opacity'] = 1.0
                    vis_data.append(mesh)

                if MESH_PRED_VIS:
                    mesh_pred_denormalized['color'] = cat2color_map[cat]
                    mesh_pred_denormalized['opacity'] = 1.0
                    vis_data.append(mesh_pred_denormalized)

                if SEGPCD_VIS:
                    seg_pcd_path = os.path.join(data_root,instance_data["segmented_cloud"])
                    seg_pcd = load_pcd_vis( seg_pcd_path, sub_sample=int( M/len(instances_data.keys()) ) )
                    seg_pcd.name = "pcd_{}".format(instance_id)
                    # seg_pcd['color'] = pcd_color
                    vis_data.append(seg_pcd)
            # break
            # pcd_meshcoord = (np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T

    layout = go.Layout(scene=dict(
            aspectmode='data',  # Set the aspect ratio to 'cube' for equal scales
            xaxis=dict(showbackground=False, showgrid=False, showline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, showline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showgrid=False, showline=False, showticklabels=False, title=''),
        ),
        showlegend=False
        )

    fig = go.Figure(data=vis_data, layout=layout)
    # fig.show()
    # pio.write_html(fig, "scene0248_00diff.html")
    pio.write_image(fig,"/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/metrics/scenes_all/{}.png".format(scene_name))
        