## Pipeline 1 : Dino_V2 encoder Modification

### Step 1: Generate the expert observations:
Either generate expert observations one experiment at a time:
``` bash scripts/gen_demonstration_metaworld.sh {task_name} ```

Or you can do batch processing of tasks by running:
``` bash scripts/run_bash_demonstrations.sh {file_to_list_of_experiments}.txt ```
Adjust the num_points and image sizes to (1024,128) or (4096,512)

### Step 2: Rename folders for the experiment
Change the names of the observations to {file_name}_dino.zarr for (4096,512) config or {file_name}_dino_128.zarr for (1024,128)
also need to change crop_shape in dp3_dino.yaml from 512 to 128 for the training.

### Step 3: Run training script
Run the training script:
``` bash scripts/train_policy.sh dp3_dino metaworld_{task_name}_dino {tags for training} {cuda device id} 0 ```

for low res, run this script:
``` bash scripts/train_policy.sh dp3_dino metaworld_{task_name}_dino_128 {tags for training} {cuda device id} 0 ```

### Step 4: Run eval script:
Run the training script:
``` bash scripts/eval_policy.sh dp3_dino metaworld_{task_name}_dino {tags for training} {cuda device id} 0 ```

for low res, run this script:
``` bash scripts/eval_policy.sh dp3_dino metaworld_{task_name}_dino_128 {tags for training} {cuda device id} 0 ```



## Pipeline 2: Depth Anything V3 adaptation
### Step 1: Generate the expert observations:
Either generate expert observations one experiment at a time:
``` bash scripts/gen_demonstration_metaworld.sh {task_name} ```

Or you can do batch processing of tasks by running:
``` bash scripts/run_bash_demonstrations.sh {file_to_list_of_experiments}.txt ```
Adjust the num_points and image sizes to (512,128) or (4096,512)

### Step 2: Rename files for experiment
Change the names of the observations to {file_name}_4096.zarr for (4096,512) config or {file_name}_128.zarr for (512,128)
also need to change crop_shape in dp3.yaml from 512 to 128 for the training.

### Step 3: run the depth_anything wrapper
Run the depth_anything to overwrite the depth values from depth_anything

python scripts/depth_anything_wrapper.py \
    --zarr_path 3D-Diffusion-Policy/data/metaworld_pick-place_expert.zarr \
    --device cuda \
    --batch_size 10 \
    --num_points 4096

### Step 3: Run training script
Run the training script, for low res:
``` bash scripts/train_policy.sh dp3_dino metaworld_{task_name}_128 {tags for training} {cuda device id} 0 ```

for high res, run this script:
``` bash scripts/train_policy.sh dp3_dino metaworld_{task_name}_dino_4096 {tags for training} {cuda device id} 0 ```

### Step 4: Run eval script:
Run the eval script, for low res:
``` bash scripts/eval_policy.sh dp3 metaworld_{task_name}_128 {tags for training} {cuda device id} 0 ```

for high res, run this script:
``` bash scripts/eval_policy.sh dp3 metaworld_{task_name}_dino_4096 {tags for training} {cuda device id} 0 ```