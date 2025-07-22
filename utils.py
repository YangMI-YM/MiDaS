"""Utils for monoDepth.
"""
import sys
from glob import glob
import os
import re
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2) # this works
import torch
from scipy.ndimage import binary_fill_holes


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def pad_to_square_np(img_cv, gray_value, dim=1024):
    h, w = img_cv.shape[:2]
    new_dim = max(h, w)
    if new_dim > dim:
        dim = new_dim
    top = (dim - h) // 2
    bottom = dim - h - top
    left = (dim - w) // 2
    right = dim - w - left

    # Pad with black (0,0,0)
    padded_img = cv2.copyMakeBorder(img_cv, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=(gray_value, gray_value, gray_value))


def pad_to_square_pil(img_rgb, gray_value, dim=1024):
    w, h = img_rgb.size
    new_dim = max(w, h)
    if new_dim > dim:
        dim = new_dim
    new_img = Image.new("RGB", (dim, dim), (gray_value, gray_value, gray_value))
    paste_x = (dim - w) // 2
    paste_y = (dim - h) // 2
    new_img.paste(img_rgb, (paste_x, paste_y))
    new_img.resize((dim, dim))
    return new_img


def read_image(path, graynish):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = Image.open(path).convert("RGBA")
    # Create a black background image
    black_bg = Image.new("RGBA", img.size, (graynish, graynish, graynish, graynish))
    #white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # Composite original image over black background using alpha channel
    img_rgb = Image.alpha_composite(black_bg, img).convert("RGB")
    img_rgb = pad_to_square_pil(img_rgb, gray_value=graynish, dim=max(img.size))
    # Convert to numpy
    img = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    #img = cv2.imread(path, cv2.UNCHANGED)
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def read_image_extended(path, graynish):
    """
        Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = Image.open(path).convert("RGBA")
    pad_dim = max(img.size) + 500
    # Create a black background image
    black_bg = Image.new("RGBA", img.size, (graynish, graynish, graynish, graynish))
    
    # Composite original image over black background using alpha channel
    img_rgb = Image.alpha_composite(black_bg, img).convert("RGB")
    img_rgb = pad_to_square_pil(img_rgb, gray_value=graynish, dim=pad_dim)
    # Convert to numpy
    img = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    #img = cv2.imread(path, cv2.UNCHANGED)
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return


def quick_mask(rgba_image_path, threshold=1):
    """
    Generate a binary mask from an RGBA image.
    
    Parameters:
        rgba_image_path (str): Path to the RGBA image.
        threshold (int): Alpha threshold to consider a pixel visible (0-255).
        
    Returns:
        mask (np.ndarray): Binary mask (uint8) with 255 for visible, 0 for transparent.
    """
    # Load image
    img = Image.open(rgba_image_path).convert('RGBA')
    
    # Convert to NumPy array
    rgba = np.array(img)
    
    # Extract alpha channel
    ##black_mask = rgba[:,:,:3]==(0,0,0) # rgb
    ##alpha = np.where(black_mask, 0, 255) # rgb
    alpha = rgba[:, :, 3] # rgba
    
    # Generate mask: 255 for visible, 0 for transparent
    mask = np.where(alpha >= threshold, 255, 0).astype(np.uint8)
    
    # Fill holes
    ##binary = (mask > 0).any(axis=2).astype(np.uint8) # rgb
    ##filled = binary_fill_holes(binary).astype(np.uint8) # rgb
    ##mask = filled * 255 # rgb

    
    return mask


if __name__ == "__main__":
    input_path = "/home/yangmi/MiDaS/input/PurpleIron/"
    output_path = "/home/yangmi/MiDaS/output/PurpleIron_mask/"
    os.makedirs(output_path, exist_ok=True)
    image_list = glob(input_path + "*.png") + glob(input_path + "*.jpg")
    image_names = [img for img in image_list if "-bl" not in img]

    for img_name in image_names:
        visible_mask = quick_mask(img_name)
        Image.fromarray(visible_mask).save(os.path.join(output_path, ''.join(os.path.basename(img_name).split('.')[:-1])+'.png'))