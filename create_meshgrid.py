import os
from glob import glob
import torch
import utils
import cv2
import argparse
import time
import json
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def create_mesh_grid(depth_map_file, interpolation_method='cubic', smoothing_sigma=1.0, cache_file=None):
    """Create 3D mesh grid using interpolation or load from cache if available."""
    # Create cache filename based on image dimensions and parameters
    cache_file = depth_map_file.replace(".png", ".npz")
    #os.makedirs(cache_dir, exist_ok=True)

    #if os.path.isfile(cache_file):
    #    return

    print(f"Creating new meshgrid and saving to cache: {cache_file}")
    depth_map = cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE)
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]
    height, width = depth_map.shape
    
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    
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

    if points.shape[0] != values.shape[0]:
        print("Warning: Mismatched dimensions detected, adjusting points array...")
        # Ensure points array matches values array
        min_dim = min(points.shape[0], values.shape[0])
        points = points[:min_dim]
        values = values[:min_dim]
    
    # Use specified interpolation method
    method = 'linear' if interpolation_method != 'cubic' else 'cubic'
    #Z = griddata(points, values, grid_points, method=method) ## duplicate!!!
    
    #print(points.shape, values.shape, grid_points.shape, Z.shape)
    #Z = Z.reshape(height, width)
    try:
        Z = griddata(points, values, grid_points, method=method)
        print(f"Interpolation output shape: {Z.shape}")
        
        # Reshape with error handling
        try:
            Z = Z.reshape(height, width)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print("Attempting to fix reshape dimensions...")
            # If reshape fails, try to pad or trim the array
            target_size = height * width
            current_size = Z.size
            if current_size < target_size:
                # Pad with mean value if too small
                pad_size = target_size - current_size
                Z = np.pad(Z, (0, pad_size), mode='mean')
            elif current_size > target_size:
                # Trim if too large
                Z = Z[:target_size]
            Z = Z.reshape(height, width)
            print(f"Fixed reshape to shape: {Z.shape}")

    except Exception as e:
        print("Falling back to bilinear interpolation...")
        Z = cv2.resize(depth_map, (width, height), 
                       interpolation=cv2.INTER_LINEAR).astype(np.float32)
    
    # Fill NaN values if any
    if np.any(np.isnan(Z)):
        nan_mask = np.isnan(Z)
        Z[nan_mask] = np.nanmean(values)
    
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
    
    #print(np.unique(Z))
    if len(np.unique(Z)) == 1:
        print(f"Warning: check {depth_map_file}!") 
    # Save to cache
    np.savez_compressed(
        cache_file,
        X=X,
        Y=Y,
        Z=Z,
        metadata=np.array([  # Store metadata for validation
            width,
            height,
            smoothing_sigma
        ])
    )


def create_meshgrid_adaptive(depth_map_file, mask_file=False, interpolation_method='cubic', smoothing_sigma=1.0, cache_file=None):
    """Create 3D mesh grid using interpolation or load from cache if available."""
    # Create cache filename based on image dimensions and parameters
    cache_file = depth_map_file.replace(".png", "_adp.npz")
    #os.makedirs(cache_dir, exist_ok=True)

    print(f"Creating new meshgrid and saving to cache: {cache_file}")
    depth_map = cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE)
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]
    height, width = depth_map.shape
    
   # Apply different strides
    dense_stride = 1
    sparse_stride = 3
    
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

    if mask_file:
        print("object mask provided!")
        mask_flat = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).ravel()
        dense_indices = np.where(mask_flat)[0][::dense_stride]
        sparse_indices = np.where(~mask_flat)[0][::sparse_stride]

        selected_indices = np.concatenate([dense_indices, sparse_indices])
        points = points[selected_indices]
        values = values[selected_indices]

    # Normalize coordinates for interpolation
    points = points / [width-1, height-1] * 2 - 1
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    if points.shape[0] != values.shape[0]:
        print("Warning: Mismatched dimensions detected, adjusting points array...")
        # Ensure points array matches values array
        min_dim = min(points.shape[0], values.shape[0])
        points = points[:min_dim]
        values = values[:min_dim]
    
    # Use specified interpolation method
    method = 'linear' if interpolation_method != 'cubic' else 'cubic'
    #Z = griddata(points, values, grid_points, method=method) ## duplicate!!!
    
    #print(points.shape, values.shape, grid_points.shape, Z.shape)
    #Z = Z.reshape(height, width)
    try:
        Z = griddata(points, values, grid_points, method=method)
        print(f"Interpolation output shape: {Z.shape}")
        
        # Reshape with error handling
        try:
            Z = Z.reshape(height, width)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print("Attempting to fix reshape dimensions...")
            # If reshape fails, try to pad or trim the array
            target_size = height * width
            current_size = Z.size
            if current_size < target_size:
                # Pad with mean value if too small
                pad_size = target_size - current_size
                Z = np.pad(Z, (0, pad_size), mode='mean')
            elif current_size > target_size:
                # Trim if too large
                Z = Z[:target_size]
            Z = Z.reshape(height, width)
            print(f"Fixed reshape to shape: {Z.shape}")

    except Exception as e:
        print("Falling back to bilinear interpolation...")
        Z = cv2.resize(depth_map, (width, height), 
                       interpolation=cv2.INTER_LINEAR).astype(np.float32)
    
    # Fill NaN values if any
    if np.any(np.isnan(Z)):
        nan_mask = np.isnan(Z)
        Z[nan_mask] = np.nanmean(values)
    
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
    
    #print(np.unique(Z))
    if len(np.unique(Z)) == 1:
        print(f"Warning: check {depth_map_file}!") 
    # Save to cache
    np.savez_compressed(
        cache_file,
        X=X,
        Y=Y,
        Z=Z,
        metadata=np.array([  # Store metadata for validation
            width,
            height,
            smoothing_sigma
        ])
    )
    
        
def clear_meshgrid_cache(self):
    """Clear all cached meshgrids."""
    cache_dir = "./meshgrid_cache"
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.startswith("meshgrid_") and file.endswith(".npz"):
                os.remove(os.path.join(cache_dir, file))
        print("Cleared meshgrid cache")


if __name__ == "__main__":
    '''
    input_path = "/home/yangmi/s3data/AutoLabel/depth-anything-v2/crV-Prods"
    # get input
    image_data = os.listdir(input_path)
    num_images = len(image_data)
    image_names = [img for img in image_data if img.endswith('-blanc.png')]

    for img_name in image_names:
        create_mesh_grid(os.path.join(input_path, img_name))'''
    prod_name = "PotDeMiel" # "PurpleIron" # 
    input_path = f"/home/yangmi/MiDaS/input/{prod_name}/"
    mask_path = f"/home/yangmi/MiDaS/output/{prod_name}_mask/"
    #image_data = glob(input_path + "*/*/*.png") + glob(input_path + "*/*.png")
    image_names = glob(input_path + "*-bl.png") #[img for img in image_data if "-" in img]

    for img_name in image_names:
        mask_name = mask_path + os.path.basename(img_name).split("-bl")[0] + ".png"
        print(mask_name)
        create_meshgrid_adaptive(img_name, mask_name)
