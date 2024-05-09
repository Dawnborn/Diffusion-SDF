import plotly.graph_objects as go
import trimesh

def load_mesh(file_path):
    """
    Load a mesh from a file. The function is compatible with various file formats,
    including OBJ and PLY, thanks to trimesh's automatic format detection.
    
    Parameters:
    - file_path: str, path to the mesh file (.obj, .ply, etc.)
    
    Returns:
    - vertices: (N, 3) ndarray of vertices.
    - faces: (M, 3) ndarray of faces.
    """
    # Load the mesh file with automatic format detection
    mesh = trimesh.load(file_path, force='mesh')
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file does not contain a valid mesh.")
    
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    return vertices, faces

def plot_mesh(vertices, faces, color):
    """
    Create a Plotly Mesh3d trace from vertices and faces.
    
    Parameters:
    - vertices: (N, 3) ndarray of vertices.
    - faces: (M, 3) ndarray of faces.
    - color: str, color of the mesh.
    
    Returns:
    - A Plotly Mesh3d trace.
    """
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5
    )
    return mesh_trace

# Example usage:
fig = go.Figure()

# Load and plot the first mesh (can be .obj or .ply)
vertices1, faces1 = load_mesh('/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/49999/Meshes/1a4a8592046253ab5ff61a3a2a0e2484_scene0484_00_ins_1.ply.ply')
fig.add_trace(plot_mesh(vertices1, faces1, 'blue'))

# Load and plot the second mesh (can be .obj or .ply)
vertices2, faces2 = load_mesh('/storage/user/huju/transferred/ws_dditnach/DATA/ScanARCW/canonical_mesh_manifoldplus/04256520/1a4a8592046253ab5ff61a3a2a0e2484_scene0484_00_ins_1/model_canonical_manifoldplus.obj')
fig.add_trace(plot_mesh(vertices2, faces2, 'red'))

# Update plot layout
fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False)
    ),
    title_text="3D Mesh Visualization"
)

fig.show()
