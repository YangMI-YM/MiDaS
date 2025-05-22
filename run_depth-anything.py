"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time
import json
import numpy as np
from image_gen_aux import DepthPreprocessor


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


def run(input_path, output_path, model_type="depth-anything-large-hf", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    """Run MonoDepthNN to compute depth maps.

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

    # Load depth-anything-large-hf
    processor = DepthPreprocessor.from_pretrained(model_type)

    # get input
    image_data = json.load(open(input_path))["mixed"]
    num_images = len(image_data)
    image_names = [img["prod_img"] for img in image_data if "hed" not in img["prod_img"]]

    multi_prod_names = set(json.load(open(input_path)).get("multi-prod"))
    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    if input_path is not None:
        if output_path is None:
            print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
        for index, image_name in enumerate(image_names):
            #if not '.'.join(os.path.basename(image_name).split('.')[:-1]) in multi_prod_names:
            #    continue
            print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

            # input
            original_image_rgb = utils.read_image(image_name)  # in [0, 1]
            # Convert to torch tensor manually (if transform expects tensor)
            input_tensor = torch.from_numpy(original_image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Apply MiDaS's expected preprocessing
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(original_image_rgb.shape[0], original_image_rgb.shape[1]), mode="bicubic", align_corners=False)

            # compute
            with torch.no_grad():
                prediction = processor(input_tensor)
                prediction = np.asarray(prediction[0])  #(np.asarray(prediction[0]) * 255).astype(np.uint8) ###overflow!!!###
            
            # output
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type.split('/')[-1]
                )
                if not side:
                    #utils.write_depth(filename, prediction, grayscale, bits=1)
                    content = create_side_by_side(None, prediction, grayscale)
                    cv2.imwrite(filename + ".png", content)
                else:
                    original_image_bgr = np.flip(original_image_rgb, 2)
                    content = create_side_by_side(original_image_bgr*255, prediction, grayscale)
                    cv2.imwrite(filename + ".png", content)
                #utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))

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
                        default="/home/yangmi/s3data/Depth_ckpts/dpt_beit_large_512.pt",
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='LiheYoung/depth-anything-large-hf',
                        help='Model type: see https://github.com/asomoza/image_gen_aux.git, '
                            'Tried: LiheYoung/depth-anything-large-hf, depth-anything/Depth-Anything-V2-Large-hf, depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf' 
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
    run(args.input_path, args.output_path, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)
