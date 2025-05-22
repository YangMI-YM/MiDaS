import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from scipy.interpolate import Rbf


def estimate_depth(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_rgb = image_rgb / 255.0  # scale to [0, 1]

    # Convert to torch tensor manually (if transform expects tensor)
    input_tensor = torch.from_numpy(input_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Apply MiDaS's expected preprocessing
    input_tensor = torch.nn.functional.interpolate(input_tensor, size=(384, 384), mode="bicubic", align_corners=False)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return normalized_depth


# Thin-Plate Spline Warping
def warp_tps(label_img, src_pts, dst_pts, shape):
    h, w = shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    flat_grid = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # RBF interpolation for X and Y coordinates
    fx = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0], function='thin_plate')
    fy = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1], function='thin_plate')

    map_x = fx(flat_grid[:, 0], flat_grid[:, 1]).reshape(h, w).astype(np.float32)
    map_y = fy(flat_grid[:, 0], flat_grid[:, 1]).reshape(h, w).astype(np.float32)

    warped = cv2.remap(label_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return warped

# Curved-aware warp and blend
def warp_and_blend(product_img, label_img, dst_pts):
    h, w = label_img.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    warped = warp_tps(label_img, src_pts, dst_pts, product_img.shape[:2])

    # Alpha blending
    if label_img.shape[2] == 4:
        alpha = warped[:, :, 3] / 255.0
        alpha = alpha[..., np.newaxis]
        overlay = warped[:, :, :3].astype(float)
        base = product_img.astype(float)
        blended = (1 - alpha) * base + alpha * overlay
        return blended.astype(np.uint8)
    else:
        return cv2.addWeighted(product_img, 0.8, warped[:, :, :3], 0.2, 0)

# ----------------------------
# Sliders for user-defined corners
# ----------------------------
def setup_corner_sliders(image_shape):
    h, w = image_shape[:2]
    return [
        (widgets.IntSlider(value=100 + 100 * (i % 2), min=0, max=w, description=f'X{i}'),
         widgets.IntSlider(value=100 + 100 * (i // 2), min=0, max=h, description=f'Y{i}'))
        for i in range(4)
    ]

def display_corner_sliders(sliders):
    for x, y in sliders:
        display(x, y)

def get_dst_corners_from_sliders(sliders):
    return [[x.value, y.value] for x, y in sliders]


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid