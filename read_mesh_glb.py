import trimesh
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_to_image import glb_to_image
from PIL import Image
import cv2
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import json
import OpenEXR

def load_exr_file(filepath):
    """Load EXR file and return RGB channels as numpy arrays."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Open EXR file
    exr_file = OpenEXR.InputFile(filepath)
    
    # Get header info
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # Read channels
    channels = ['R', 'G', 'B']
    pixel_data = []
    
    for channel in channels:
        if channel in header['channels']:
            # Read channel data
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channel_data = array.array('f', exr_file.channel(channel, FLOAT)).tolist()
            pixel_data.append(channel_data)
        else:
            # If channel doesn't exist, create zero array
            pixel_data.append([0.0] * (size[0] * size[1]))
    
    exr_file.close()
    
    # Reshape and stack channels
    pixel_data = [np.array(channel).reshape(size[1], size[0]) for channel in pixel_data]
    result = np.stack(pixel_data, axis=2)
    
    return result

def render_glb_to_png(glb_filepath: str, output_png: str):
    with open(glb_filepath, 'rb') as f:
        glb_bytes = f.read()
    img_bytes = glb_to_image(glb_bytes)
    if hasattr(img_bytes, 'getvalue'):
        with open(output_png, 'wb') as out:
            out.write(img_bytes.getvalue())
    else:
        raise RuntimeError(f"Rendering failed: {img_bytes}")
    
def visualize_vertices(vertices: np.ndarray, save_path: str = None):
    """
    Visualize 3D vertices as a scatter plot.
    
    Parameters:
        vertices (np.ndarray): shape (N, 3), 3D coordinates
        save_path (str): Optional path to save the rendered .png
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = vertices[::100, 0], vertices[::100, 1], vertices[::100, 2]
    ax.scatter(x, y, z, c='blue', s=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Vertices Visualization")
    ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def vertices_from_depth(depth_map_file, interpolation_method='cubic', smoothing_sigma=1.0):
    depth_map = cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE)
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]
    height, width = depth_map.shape

    # Create base coordinate grid
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    # Create structured grid points
    y_coords = np.arange(height)
    x_coords = np.arange(width)
    X_coords, Y_coords = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X_coords.ravel(), Y_coords.ravel()))
    values = depth_map.ravel()
    
    # Normalize coordinates for interpolation
    points = points / [width-1, height-1] * 2 - 1
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    Z = griddata(points, values, grid_points, method=interpolation_method)
    # Fill NaN values if any
    if np.any(np.isnan(Z)): 
        nan_mask = np.isnan(Z)
        #Z[nan_mask] = np.nanmean(values)
        Z[nan_mask] = griddata(points, values, grid_points[nan_mask], method=interpolation_method)
    #rbf = Rbf(points[:, 0], points[:, 1], values, function='thin_plate')  # or 'thin_plate', 'multiquadric'
    #Z = rbf(grid_points[:, 0], grid_points[:, 1])
    
    # Apply additional smoothing
    kernel_size = max(3, int(2 * smoothing_sigma + 1))
    if kernel_size % 2 == 0:  # Ensure odd kernel size
        kernel_size += 1
    if smoothing_sigma > 0:
        Z = cv2.GaussianBlur(Z.astype(np.float32), 
                             (kernel_size, kernel_size), 
                             smoothing_sigma)
    
    # Ensure proper range
    Z = np.clip(Z, 0, 255).astype(np.float32)
    
    return Z

def visualize_depth_anything(depth_img_pth, grid_data, save_path):
    img = np.array(Image.open(depth_img_pth).convert("L"))
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1) # depth vis in grayscale
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255); plt.title("Depth vis"); plt.axis("off")

    H, W = img.shape
    x_cols, y_rows = np.meshgrid(np.arange(W), np.arange(H))  # create a meshgrid
    # Normalize to [0, 1] to get UVs
    u = x_cols / (W - 1)
    v = y_rows / (H - 1)
    plt.subplot(1, 3, 2) # UVs as scatter plot. If points are uniformly aligned in a grid, spacing is likely regular.
    plt.scatter(u, v, s=1)
    plt.title("UV Coordinates")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.gca().invert_yaxis()  # Optional: image-like orientation
    plt.grid(True)

    grid_z = np.load(grid_data)['Z']
    plt.subplot(1, 3, 3) # Visualize interpolated depth
    plt.imshow(grid_z, cmap='hot')
    plt.title("Interpolated Z from UV → Depth Map")
    plt.colorbar(label='Z (depth)')
    plt.axis('off')
    plt.savefig(save_path, transparent=True, dpi=300)
    plt.close()


def visualize(depth_img_pth, mesh_uvs, mesh_vertices, mask_pth, save_path):
    img = np.array(Image.open(depth_img_pth).convert("L"))
    mask = np.array(Image.open(mask_pth).convert("L")).astype(bool)
    plt.figure(figsize=(22, 6))

    plt.subplot(1, 4, 1) # depth vis in grayscale
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255); plt.title("Depth vis"); plt.axis("off")

    plt.subplot(1, 4, 2) # UVs as scatter plot. If points are uniformly aligned in a grid, spacing is likely regular.
    plt.scatter(mesh_uvs[:, 0], mesh_uvs[:, 1], s=1)
    plt.title("UV Coordinates")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.gca()#.invert_yaxis()  # Optional: image-like orientation
    plt.grid(True)

    plt.subplot(1, 4, 3) # Visualize interpolated depth
    height, width = img.shape
    points = mesh_uvs * [width - 1, height - 1]  # UV in pixel space
    values = mesh_vertices[:, 2]          # Z values, or any per-vertex scalar

    # Create regular grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, width - 1, width),
        np.linspace(0, height - 1, height)
    )
    # Interpolate Z onto the grid
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    #print(grid_z[0][0], "nan") 
    grid_z[~np.flipud(mask)] = np.nan

    #plt.imshow(np.flipud(grid_z), cmap='hot')
    plt.imshow(grid_z, cmap='hot', origin='lower')
    plt.title("Interpolated Z from UV → Depth Map")
    plt.colorbar(label='Z (depth)')
    plt.axis('off')

    plt.subplot(1, 4, 4) # depthmap-oriented
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create structured grid points
    y_coords = np.arange(height)
    x_coords = np.arange(width)
    X_coords, Y_coords = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X_coords.ravel(), Y_coords.ravel()))
    values = img.ravel() # load_exr_file(os.path.join(os.path.dirname(depth_img_pth), "depth.exr")) np.unique(values)=0 !
    
    # Normalize coordinates for interpolation
    points = points / [width-1, height-1] * 2 - 1
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    Z = griddata(points, values, grid_points, method="cubic")
    Z = Z.reshape(height, width)
    Z[~mask] = 0
    plt.imshow(Z, cmap='hot')
    plt.title("Interpolated Z from depth vis")
    plt.colorbar(label='Z (depth)')
    plt.axis('off')
    plt.savefig(save_path, transparent=True, dpi=300)
    plt.close()


# Load GLB file using trimesh ## save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)
mask_pth = "/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/mask.png"
mesh_example = "/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/mesh.glb" # for 3D interaction
depth_vis_example = "/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/depth_vis.png" # scale invariant (d up to an unknown metric scale factor, resulting in d=s*p+t)
mesh_scene = trimesh.load(mesh_example, force='scene') # Make sure it loads as a scene
print("scene attributes", mesh_scene.geometry.keys()) 
# Get the first mesh from the scene
mesh = list(mesh_scene.geometry.values())[0] 
print(f"Loaded object type: {type(mesh)}.") # <class 'trimesh.base.Trimesh'>
print(f"Object attributes: {dir(mesh)}.") 
# Access vertices and faces
vertices = mesh.vertices        # (N, 3) float32    (1830364, 3) 
faces = mesh.faces              # (M, 3) int32      (3654236, 3)
uvs = mesh.visual.uv            # (N, 2): UV coordinates    (1830364, 2)

#render_glb_to_png(mesh_example, mesh_example.replace(".glb", ".png"))
#visualize_vertices(vertices, save_path="./vertices.png") # mesh_example.replace("mesh.glb", "vertices.png")
#depth_Z = vertices_from_depth(depth_vis_example)
visualize(depth_vis_example, uvs, vertices, mask_pth, save_path="./sbys.png")

depth_map_dav2 = "/home/yangmi/s3data/AutoLabel/depth-anything-v2/PotDeMiel/小蜜罐-俯4-bl.png"
depth_griddata = "/home/yangmi/MiDaS/input/PotDeMiel/小蜜罐-俯4-bl.npz"
visualize_depth_anything(depth_map_dav2, depth_griddata, save_path="./no_mask.png")

            