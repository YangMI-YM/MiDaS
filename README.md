## Depth estimation & meshgrid generation

This repo branch contains the implementations of 
- Depth map estimation using depth-anything. 
- Meshgrid generation.


### Depth map 

#### Setup 


1) Installation: 

    ```shell
    pip install -q opencv-python torch torchvision matplotlib timm scipy ipywidgets
    ```

If encounters any dependencies conflict, please refer to pip_freeze.txt for full list of package versions.

    
#### Usage

1) Place one or more input images in your input folder.

2) Run the model with

   ```shell
   CUDA_VISIBLE_DEVICES=$GPU_ID python run_depth-anything.py --input $INPUT --output_path $OUTPUT --model_type $model_type --grayscale
   ```
   where ```<model_type>``` is chosen from [depth-anything/Depth-Anything-V2-Large-hf](#model_type), [depth-anything/Depth-Anything-V2-Large](#model_type),
   [depth-anything/Depth-Anything-V2-Small-hf](#model_type), [depth-anything/Depth-Anything-V2-Small](#model_type), [depth-anything/Depth-Anything-V2-Base-hf](#model_type),
   [depth-anything/Depth-Anything-V2-Base](#model_type), [LiheYoung/depth-anything-large-hf](#model_type), [LiheYoung/depth-anything-large](#model_type),
   [LiheYoung/depth-anything-small-hf](#model_type), [LiheYoung/depth-anything-small](#model_type), [LiheYoung/depth-anything-base-hf](#model_type),
   [LiheYoung/depth-anything-base](#model_type), [depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf](#model_type).
 
3) The resulting grayscale depth maps are written to the `$OUTPUT` folder. Please be aware that the input images will be padded and resized to 1024x1024 (or 512x512, can be adjusted at request) for the trade-off depthmap quality and processing speed. 



### Meshgrid

#### Usage

1) Set input depth map path, e.g., $INPUT_DEPTH_MAP=$INPUT_DIR/$INPUT_FILENAME.


2) Call ```create_mesh_grid($INPUT_DEPTH_MAP)``` in ```create_meshgrid.py```.


3) The resulting meshgrids are stored as `.npz` files under the directory of depthmaps. 






