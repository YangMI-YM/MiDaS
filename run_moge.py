"""Compute depth maps for images in the input folder.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from glob import glob
import torch
import utils
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata
import argparse
import time
import json
import numpy as np
from image_gen_aux import DepthPreprocessor

import trimesh
import trimesh.visual
import click

from moge.model import import_model_class_by_version
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
import utils3d


first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale, norm=True):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """

    if norm:
        right_side = np.repeat(np.expand_dims(depth, 2), 3, axis=2) #/ 3
    else:
        right_side = np.repeat(np.expand_dims(depth, 2), 3, axis=2) / 3
    
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


def run_moge(input_path, output_path, model_type="v2", optimize=False, side=False, height=None,
        square=False, grayscale=False, use_fp16=True):
    """Run MonoDepthNN to compute depth maps.
    Terms     |   Description 
    points    |   (H, W, 3) or (N, 3) array of 3D coordinates. Usually a depth meshgrid converted to world/camera space.
    vertices  |   Nx3. Each row represents a point (x, y, z) in 3D space that defines the shape of the mesh.
    faces	  |   Mx3. Each row lists indices into the vertices array, forming a triangle.
    vertex_uvs|	  Nx2. Array of 2D texture coordinates (u, v) that map each vertex to a pixel in the texture image. Used to apply the texture (color_texture) to the 3D surface.
    material  |	  Describes how the mesh should look when rendered (color texture, normal maps, metalness, roughness).

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Load MOGE2
    DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal", # full capacity
    }
    pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_type]
    model = import_model_class_by_version(model_type).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16:
        model.half()

    # get input
    image_data = glob(input_path+"/*/*.png") + glob(input_path+"/*/*/*.png") #glob(input_path+"/*.png") + glob(input_path+"/*.jpg") #
    image_names = [img_name for img_name in image_data if "composed" not in img_name and "bl" not in img_name]
        
    num_images = len(image_data)
    print(image_names)

    # output options
    save_maps_ = save_glb_ = save_ply_ = save_npz_ = True
    show = False
    fov_x_ = None
    resolution_level = 9
    num_tokens = None
    
    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    if input_path is not None:
        if output_path is None:
            print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
        for index, image_name in enumerate(image_names):
            #if not '.'.join(os.path.basename(image_name).split('.')[:-1]) in multi_prod_names:
            #if os.path.isfile(os.path.join(output_path, os.path.basename(image_name))):
            #    continue
            print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))
            # input
            img = Image.open(image_name).convert("RGBA")
            # Create a black background image
            black_bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
            # Composite original image over black background using alpha channel
            img_rgb = Image.alpha_composite(black_bg, img).convert("RGB")
            # Convert to numpy
            image = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            #image = cv2.cvtColor(cv2.imread(str(image_name)), cv2.COLOR_BGR2RGB)
            image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

            # Inference
            output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
            points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
            normal = output['normal'].cpu().numpy() if 'normal' in output else None
            save_path = Path("".join(image_name.split(".")[:-1]).replace(input_path, output_path))
            #save_path = Path(output_path, "".join(os.path.basename(image_name).split(".")[:-1]))
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Save images / maps
            if save_maps_:
                cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / 'depth.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
                cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                if normal is not None:
                    cv2.imwrite(str(save_path / 'normal_vis.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_path / 'normal.exr'), normal, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
                with open(save_path / 'fov.json', 'w') as f:
                    json.dump({
                        'fov_x': round(float(np.rad2deg(fov_x)), 2),
                        'fov_y': round(float(np.rad2deg(fov_y)), 2),
                    }, f)

            # Export mesh & visulization
            if save_glb_ or save_ply_ or show:
                mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=0.04)
                if normal is None:
                    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                        points,
                        image.astype(np.float32) / 255,
                        utils3d.numpy.image_uv(width=width, height=height),
                        mask=mask_cleaned,
                        tri=True
                    )
                    vertex_normals = None
                else:
                    faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                        points,
                        image.astype(np.float32) / 255,
                        utils3d.numpy.image_uv(width=width, height=height),
                        normal,
                        mask=mask_cleaned,
                        tri=True
                    )
                # When exporting the model, follow the OpenGL coordinate conventions:
                # - world coordinate system: x right, y up, z backward.
                # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
                vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
                if normal is not None:
                    vertex_normals = vertex_normals * [1, -1, -1]

            if save_glb_:
                save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)

            if save_ply_:
                save_ply(save_path / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

            if save_npz_:
                height, width, _ = image.shape
                points = vertex_uvs * [width - 1, height - 1]  # UV in pixel space
                values = vertices[:, 2]          # Z values, or any per-vertex scalar

                # Create regular grid
                grid_x, grid_y = np.meshgrid(
                    np.linspace(0, width - 1, width),
                    np.linspace(0, height - 1, height)
                )
                # Interpolate Z onto the grid
                grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
                #print(grid_z[0][0], "nan") 
                grid_z[~np.flipud(mask)] = np.nan

                np.savez_compressed(save_path / 'vertices_cache.npz', X=grid_x,Y=grid_y,Z=grid_z,metadata=np.array([width, height]))

                x = np.linspace(-1, 1, width)
                y = np.linspace(-1, 1, height)
                X, Y = np.meshgrid(x, y)
                
                # Create structured grid points
                y_coords = np.arange(height)
                x_coords = np.arange(width)
                X_coords, Y_coords = np.meshgrid(x_coords, y_coords)
                points = np.column_stack((X_coords.ravel(), Y_coords.ravel()))
                values = cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2GRAY).ravel()

                # Normalize coordinates for interpolation
                points = points / [width-1, height-1] * 2 - 1
                grid_points = np.column_stack((X.ravel(), Y.ravel()))

                Z = griddata(points, values, grid_points, method="cubic")
                Z = Z.reshape(height, width)
                Z[~mask] = 255

                np.savez_compressed(save_path / 'depth_cache.npz', X=X,Y=Y,Z=Z,metadata=np.array([width, height]))

                fig, axes = plt.subplots(1, 2, figsize=(16, 12))
                plt.subplot(1, 2, 1)
                plt.imshow(grid_z, cmap='hot', origin='lower')
                plt.title("Interpolated Z from UV → Depth Map")
                plt.colorbar(label='Z (depth)')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(Z, cmap='hot')
                plt.title("Interpolated Z from depth vis")
                plt.colorbar(label='Z (depth)')
                plt.axis('off')
                plt.savefig(save_path / 'InterpolatedZ.png', transparent=True, dpi=300)
                plt.close()

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default="/home/yangmi/AutoLabel/complete_testset.json",
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default="/home/yangmi/s3data/AutoLabel",
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='LiheYoung/depth-anything-large-hf',
                        help='Model type: see https://github.com/microsoft/MoGe.git, '
                            'Tried: Ruicheng/moge-2-vitl, Ruicheng/moge-2-vitl-normal. Note: moge-2-vitl-normal has full capabilities' 
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )

    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    # CUDA_VISIBLE_DEVICES=0 python run_moge.py --input required --output_path required --model_type "v2" --grayscale
    run_moge(args.input_path, args.output_path, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)
