# OccNeRF: Rendering Humans from Object-Occluded Monocular Videos, ICCV 2023
Project Page: https://cs.stanford.edu/~xtiange/projects/occnerf/  
Paper: https://arxiv.org/pdf/2308.04622.pdf

![framework](./teaser.png)

## Prerequisite

### `Configure environment`

1. Create and activate a virtual environment.

    conda create --name occnerf python=3.7
    conda activate occnerf

2. Install the required packages as required in HumanNeRF.

    pip install -r requirements.txt

3. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) by following its official instructions.

4. Build `gridencoder` by following the instructions provided in [torch-ngp](https://github.com/ashawkey/torch-ngp/tree/main#build-extension-optional)


### `Download SMPL model`

Download the gender neutral SMPL model from [here](https://smplify.is.tue.mpg.de/), and unpack **mpips_smplify_public_v2.zip**.

Copy the smpl model.

    SMPL_DIR=/path/to/smpl
    MODEL_DIR=$SMPL_DIR/smplify_public/code/models
    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.


## Run on ZJU-Mocap Dataset

Below we take the subject 387 as a running example.

### `Prepare a dataset`

First, download ZJU-Mocap dataset from [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset). 

Second, modify the yaml file of subject 387 at `tools/prepare_zju_mocap/387.yaml`. In particular,  `zju_mocap_path` should be the directory path of the ZJU-Mocap dataset.

```
dataset:
    zju_mocap_path: /path/to/zju_mocap
    subject: '387'
    sex: 'neutral'

...
```
    
Finally, run the data preprocessing script.

    cd tools/prepare_zju_mocap
    python prepare_dataset.py --cfg 387.yaml
    cd ../../

### `Train`

To train a model: 

    python train.py --cfg configs/occnerf/zju_mocap/387/occnerf.yaml

Alternatively, you can update the scripts: in `train.sh`.

### `Render output`

Render the frame input (i.e., observed motion sequence).

    python run.py \
        --type movement \
        --cfg configs/occnerf/zju_mocap/387/occnerf.yaml 

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py \
        --type freeview \
        --cfg configs/occnerf/zju_mocap/387/occnerf.yaml \
        freeview.frame_idx 128


Render the learned canonical appearance (T-pose).

    python run.py \
        --type tpose \
        --cfg configs/occnerf/zju_mocap/387/occnerf.yaml 

In addition, you can find the rendering scripts in `scripts/zju_mocap`.


<!-- ## Run on a Custom Monocular Video

To get the best result, we recommend a video clip that meets these requirements:

- The clip has less than 600 frames (~20 seconds).
- The human subject shows most of body regions (e.g., front and back view of the body) in the clip.

### `Prepare a dataset`

To train on a monocular video, prepare your video data in `dataset/wild/monocular` with the following structure:

    monocular
        ├── images
        │   └── ${item_id}.png
        ├── masks
        │   └── ${item_id}.png
        └── metadata.json

We use `item_id` to match a video frame with its subject mask and metadata. An `item_id` is typically some alphanumeric string such as `000128`.

#### **images**

A collection of video frames, stored as PNG files.

#### **masks**

A collection of subject segmentation masks, stored as PNG files.

#### **metadata.json**

This json file contains metadata for video frames, including:

- human body pose (SMPL poses and betas coefficients)
- camera pose (camera intrinsic and extrinsic matrices). We follow [OpenCV](https://learnopencv.com/geometry-of-image-formation/) camera coordinate system and use [pinhole camera model](https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/CV/PinholeCamera/PinholeCamera.html).

You can run SMPL-based human pose detectors (e.g., [SPIN](https://github.com/nkolot/SPIN), [VIBE](https://github.com/mkocabas/VIBE), or [ROMP](https://github.com/Arthur151/ROMP)) on a monocular video to get body poses as well as camera poses. 


```javascript
{
  // Replace the string item_id with your file name of video frame.
  "item_id": {
        // A (72,) array: SMPL coefficients controlling body pose.
        "poses": [
            -3.1341, ..., 1.2532
        ],
        // A (10,) array: SMPL coefficients controlling body shape. 
        "betas": [
            0.33019, ..., 1.0386
        ],
        // A 3x3 camera intrinsic matrix.
        "cam_intrinsics": [
            [23043.9, 0.0,940.19],
            [0.0, 23043.9, 539.23],
            [0.0, 0.0, 1.0]
        ],
        // A 4x4 camera extrinsic matrix.
        "cam_extrinsics": [
            [1.0, 0.0, 0.0, -0.005],
            [0.0, 1.0, 0.0, 0.2218],
            [0.0, 0.0, 1.0, 47.504],
            [0.0, 0.0, 0.0, 1.0],
        ],
  }

  ...

  // Iterate every video frame.
  "item_id": {
      ...
  }
}
```

Once the dataset is properly created, run the script to complete dataset preparation.

    cd tools/prepare_wild
    python prepare_dataset.py --cfg wild.yaml
    cd ../../

### `Train a model`

Now we are ready to lanuch a training. By default, we used 4 GPUs (NVIDIA RTX 2080 Ti) to train a model. 

    python train.py --cfg configs/human_nerf/wild/monocular/adventure.yaml

For sanity check, we provide a single-GPU (NVIDIA RTX 2080 Ti) training config. Note the performance is not guaranteed for this configuration.

    python train.py --cfg configs/human_nerf/wild/monocular/single_gpu.yaml

### `Render output`

Render the frame input (i.e., observed motion sequence).

    python run.py \
        --type movement \
        --cfg configs/human_nerf/wild/monocular/adventure.yaml 

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py \
        --type freeview \
        --cfg configs/human_nerf/wild/monocular/adventure.yaml \
        freeview.frame_idx 128


Render the learned canonical appearance (T-pose).

    python run.py \
        --type tpose \
        --cfg configs/human_nerf/wild/monocular/adventure.yaml 

In addition, you can find the rendering scripts in `scripts/wild`. -->

## Acknowledgement

The implementation is based on [HumanNeRF](https://github.com/chungyiweng/humannerf/tree/main), which took reference from [Neural Body](https://github.com/zju3dv/neuralbody), [Neural Volume](https://github.com/facebookresearch/neuralvolumes), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), and [YACS](https://github.com/rbgirshick/yacs). We thank the authors for their generosity to release code.

## Citation

If you find our work useful, please consider citing:

```BibTeX
@InProceedings{Xiang_2023_OccNeRF,
    author    = {Xiang, Tiange and Sun, Adam and Wu, Jiajun and Adeli, Ehsan and Li, Fei-Fei},
    title     = {Rendering Humans from Object-Occluded Monocular Videos},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023}
}
```