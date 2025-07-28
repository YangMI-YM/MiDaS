import trimesh
import numpy as np

# Load GLB file using trimesh
mesh = trimesh.load("/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/mesh.glb")
print(f"Loaded object type: {type(mesh)}")
print(f"Object attributes: {dir(mesh)}")
# Extract vertex positions (X, Y, Z coordinates)
vertices = mesh.vertices
print(f"GLB depth map loaded: {vertices.shape}")
            