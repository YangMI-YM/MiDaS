import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import OpenEXR
import Imath
import array
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN
import os

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
    pixel_data = {}
    
    for channel in channels:
        if channel in header['channels']:
            # Read channel data
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channel_data = array.array('f', exr_file.channel(channel, FLOAT)).tolist()
            pixel_data[channel] = np.array(channel_data).reshape(size[1], size[0])
        else:
            # If channel doesn't exist, create zero array
            pixel_data[channel] = np.zeros((size[1], size[0]))
    
    exr_file.close()
    
    return pixel_data

def load_mask_file(filepath):
    """Load mask image and return as boolean array."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load mask image
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask image: {filepath}")
    
    # Convert to boolean (True where mask is white/non-zero)
    mask_bool = mask.astype(bool)
    
    return mask_bool

def load_image(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    img = cv2.imread(filepath)
    return img

def classify_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_vertices = len(approx)

    if num_vertices == 3:
        shape_type = "triangle"
    elif num_vertices == 4:
        shape_type = "rectangle"
    elif num_vertices > 5:
        shape_type = "ellipse"
    else:
        shape_type = "unidentified"

    return shape_type, approx

def scale_polygon(polygon, image_shape, scale=1.75, max_iter=20, shrink_step=0.95):
    # polygon: (N, 1, 2) or (N, 2)
    # fully-contained, zoomed, 4-vertrice (quadrilateral)
    if polygon.ndim == 3:
        polygon = polygon.squeeze(1)

    # Step 1: create mask from original polygon
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Step 2: get min area rect
    rect = cv2.minAreaRect(polygon.astype(np.float32))  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect).astype(np.int32)

    # Step 3: check containment and shrink if needed
    for _ in range(max_iter):
        test_mask = np.zeros_like(mask)
        cv2.fillPoly(test_mask, [box.astype(np.int32)], 255)

        # Check if box is fully inside the original polygon mask
        inside = cv2.bitwise_and(mask, test_mask)
        area_test = cv2.countNonZero(test_mask)
        area_inside = cv2.countNonZero(inside)

        if area_inside == area_test:
            centroid = np.mean(box, axis=0)
            scaled = centroid + scale * (box - centroid)
            return scaled.astype(np.int32)  # box is fully inside

        # Shrink the rectangle around center
        centroid = np.mean(box, axis=0)
        box = centroid + shrink_step * (box - centroid)
    
    centroid = np.mean(box, axis=0)
    scaled = centroid + scale * (box - centroid)
    return scaled.astype(np.int32)

def extract_roi_data(contours, image_shape):
    results = []
    for cnt in contours:
        shape_type, polygon = classify_shape(cnt)
        #x, y, w, h = cv2.boundingRect(cnt)
        zoomed_polygon = scale_polygon(polygon, image_shape, scale=0.75)
        results.append({
            "shape": shape_type,
            #"bbox": [int(x), int(y), int(w), int(h)],
            "zoomed_polygon": zoomed_polygon,
            "polygon": polygon.squeeze().tolist() if polygon.ndim == 3 else []
        })
    return results

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def detect_contours(masked_img, min_area=1e4):
    sharpen = sharpen_image(masked_img.astype(np.uint8))
    gray = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def split_surface_regions(normals, mask, threshold=0.0):
    """
    Split surface normal regions hierarchically: first by Y direction, then each Y region by X direction.
    Only considers masked regions.
    
    Args:
        normals: Dictionary with 'R', 'G', 'B' channels (z, y, x normals)
        mask: Boolean mask where True indicates valid regions to consider
        threshold: Threshold for direction change detection
    
    Returns:
        Dictionary containing hierarchical region masks
    """
    regions = {}
    valid_mask = mask.astype(bool)
    
    # Step 1: Split by Y direction (G channel) first
    y_regions = np.zeros_like(normals['G'], dtype=int)
    y_regions[valid_mask & (normals['G'] < -threshold)] = 1      # Negative Y
    y_regions[valid_mask & (normals['G'] > threshold)] = 2       # Positive Y
    y_regions[valid_mask & (np.abs(normals['G']) <= threshold)] = 3  # Near zero Y
    regions['y'] = y_regions
    
    # Step 2: For each Y region, split by X direction (R channel)
    regions['hierarchical'] = {}
    
    # Get unique Y region labels (excluding 0 which is background)
    y_labels = np.unique(y_regions)
    y_labels = y_labels[y_labels > 0]  # Remove background label
    
    for y_label in y_labels:
        y_mask = (y_regions == y_label)
        
        # Create X regions within this Y region
        x_regions = np.zeros_like(normals['B'], dtype=int)
        x_regions[y_mask & (normals['B'] < -threshold)] = 1      # Negative X
        x_regions[y_mask & (normals['B'] > threshold)] = 2       # Positive X
        x_regions[y_mask & (np.abs(normals['B']) <= threshold)] = 3  # Near zero X
        
        regions['hierarchical'][f'y{y_label}'] = x_regions
    
    return regions

def find_largest_regions(regions, mask, n_largest=3):
    """
    Find the largest connected regions for hierarchical regions, considering only masked regions.
    Filters out regions that match the entire mask area as they are not meaningful.
    
    Args:
        regions: Dictionary of region masks (including hierarchical structure)
        mask: Boolean mask where True indicates valid regions to consider
        n_largest: Number of largest regions to find
    
    Returns:
        Dictionary containing largest regions for each direction
    """
    largest_regions = {}
    mask_area = np.sum(mask)
    
    # Handle Y direction regions
    if 'y' in regions:
        y_regions = regions['y']
        masked_regions = y_regions.copy()
        masked_regions[~mask] = 0  # Set non-masked regions to 0
        
        # Get unique region values (excluding 0 which is background)
        unique_values = np.unique(masked_regions)
        unique_values = unique_values[unique_values > 0]  # Remove background
        
        if len(unique_values) > 0:
            # Calculate area of each region and filter out mask-sized regions
            region_areas = []
            for region_value in unique_values:
                area = np.sum(masked_regions == region_value)
                # Skip regions that are too close to the entire mask area (within 1% tolerance)
                if area < mask_area * 0.99:
                    region_areas.append((region_value, area))
            
            # Sort by area and get largest n
            region_areas.sort(key=lambda x: x[1], reverse=True)
            largest_n = region_areas[:n_largest]
            
            # Create masks for largest regions
            largest_masks = []
            for region_value, _ in largest_n:
                region_mask = (masked_regions == region_value).astype(np.uint8)
                largest_masks.append(region_mask)
            
            largest_regions['y'] = largest_masks
        else:
            largest_regions['y'] = []
    
    # Handle hierarchical regions (X within Y)
    if 'hierarchical' in regions:
        largest_regions['hierarchical'] = {}
        
        for y_key, x_regions in regions['hierarchical'].items():
            masked_regions = x_regions.copy()
            masked_regions[~mask] = 0  # Set non-masked regions to 0
            
            # Get unique region values (excluding 0 which is background)
            unique_values = np.unique(masked_regions)
            unique_values = unique_values[unique_values > 0]  # Remove background
            
            if len(unique_values) > 0:
                # Calculate area of each region and filter out mask-sized regions
                region_areas = []
                for region_value in unique_values:
                    area = np.sum(masked_regions == region_value)
                    # Skip regions that are too close to the entire mask area (within 1% tolerance)
                    if area < mask_area * 0.99:
                        region_areas.append((region_value, area))
                
                # Sort by area and get largest n
                region_areas.sort(key=lambda x: x[1], reverse=True)
                largest_n = region_areas[:n_largest]
                
                # Create masks for largest regions
                largest_masks = []
                for region_value, _ in largest_n:
                    region_mask = (masked_regions == region_value).astype(np.uint8)
                    largest_masks.append(region_mask)
                
                largest_regions['hierarchical'][y_key] = largest_masks
            else:
                largest_regions['hierarchical'][y_key] = []
    
    return largest_regions

def find_corner_points(normals, regions, mask):
    """
    find corner points hierachically.
    """
    # get regions
    if 'hierarchical' in regions and regions['hierarchical']:
        all_x_regions = []
        combined_x_canvas = np.zeros_like(mask, dtype=np.uint8)
        for y_key, x_regions in regions['hierarchical'].items():
            # Add X regions to the combined canvas with different labels
            x_regions_masked = x_regions.copy()
            x_regions_masked[~mask] = 0
            
            # Get unique X region values (excluding 0 which is background)
            unique_x_values = np.unique(x_regions_masked)
            unique_x_values = unique_x_values[unique_x_values > 0]
            
            for region_value in unique_x_values:
                region_mask = (x_regions_masked == region_value).astype(np.uint8)
                all_x_regions.append(region_mask)
                combined_x_canvas += region_mask * len(all_x_regions)

    # find contours in all regions
    if all_x_regions:
        all_contours = []
        contour_areas = []

        for x_region in all_x_regions:
            contours, _ = cv2.findContours(x_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter out very small contours
                    all_contours.append(contour)
                    contour_areas.append(area)
        # Sort contours by area and get the 2 largest
        if contour_areas:
            sorted_indices = np.argsort(contour_areas)[::-1]  # Descending order
            largest_2_indices = sorted_indices[:2]
            detections = []
            # find contours in original normal
            for i, idx in enumerate(largest_2_indices):
                contour = all_contours[idx]
                area = contour_areas[idx]
                # draw contour mask
                contour_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                cv2.drawContours(contour_mask, [cv2.convexHull(contour)], -1, 255, -1)
                
                normal_seg = np.stack([normals['R'], normals['G'], normals['B']], axis=2)
                normal_seg = ((normal_seg + 1) / 2) * 255 # normalize to [0, 255]
                normal_seg[contour_mask==0] = 0
                #img_seg = image.copy()
                #img_seg[contour_mask==0] = 0
                #cv2.imwrite(f"./img_seg{i}.png", img_seg)
                normal_cnts = detect_contours(normal_seg)
                if normal_cnts:
                    detections += extract_roi_data(normal_cnts, mask.shape)
    
    print(largest_2_indices, detections)
    return combined_x_canvas, detections

def visualize_regions(normals, x_comb, pinpoints):
    """
    Visualize the original normals, mask, and hierarchical regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 12))
    fig.suptitle('Surface Region Analysis', fontsize=16)

    # Original normal maps
    axes[0].imshow(normals['B'], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('X Normal (Ch=B)')
    axes[0].axis('off')
    
    # Mask visualization
    # Combined normals visualization
    combined_normals = np.stack([normals['B'], normals['G'], normals['R']], axis=2)
    combined_normals = ((combined_normals + 1) / 2) * 255
    
    pinpts_normals = combined_normals.copy()
    pinpts_normals = pinpts_normals.astype(np.uint8)
    for pinpoint in pinpoints:
        cv2.polylines(pinpts_normals, [np.array(pinpoint["polygon"], dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.polylines(pinpts_normals, [pinpoint["zoomed_polygon"]], isClosed=True, color=(255, 0, 0), thickness=2)
        #cv2.drawContours(pinpts_normals, [np.array(pinpoint["polygon"], dtype=np.int32)], -1, (255, 0, 0), 2)

    axes[1].imshow(pinpts_normals)
    axes[1].set_title('Normal, Plane, Pinpoints')
    axes[1].axis('off')

    axes[2].imshow(x_comb, cmap='Reds', alpha=0.8)
    axes[2].set_title('Regions with Color Labels')
    axes[2].axis('off')

    plt.tight_layout()
    # Save the plot as PNG
    output_filename = 'surface_region_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {output_filename}")

def main():
    """Main function to run the surface split analysis."""
    # File path
    #img = load_image("/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/image.jpg")
    normals = load_exr_file("/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/normal.exr")
    mask = load_mask_file("/home/yangmi/s3data/AutoLabel/MoGe-2/PotDeMiel/小蜜罐-俯4/mask.png")

    if mask.shape != normals['R'].shape:
        raise ValueError(f"Mask shape {mask.shape} does not match normal map shape {normals['R'].shape}")
    
    total_pixels = mask.size
    valid_pixels = np.sum(mask)
    mask_coverage = (valid_pixels / total_pixels) * 100

    regions = split_surface_regions(normals, mask, threshold=0.1)
    #largest_regions = find_largest_regions(regions, mask, n_largest=3)
    combined_x_canvas, likely_pinpts = find_corner_points(normals, regions, mask)

    visualize_regions(normals, combined_x_canvas, likely_pinpts)

if __name__ == "__main__":
    main() 