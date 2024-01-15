# OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning
### [Project Page](https://oceanying.github.io/OmniSeg3D/) | [Arxiv Paper](https://arxiv.org/abs/2311.11666)

[OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning](https://arxiv.org/abs/2311.11666)  
[Haiyang Ying](https://oceanying.github.io/)<sup>1</sup>, Yixuan Yin<sup>1</sup>, Jinzhi Zhang<sup>1</sup>, Fan Wang<sup>2</sup>, Tao Yu<sup>1</sup>, Ruqi Huang<sup>1</sup>, [Lu Fang](http://www.luvision.net/)<sup>1</sup>   
<sup>1</sup>Tsinghua Univeristy &emsp; <sup>2</sup>Alibaba Group.  


![image](https://github.com/THU-luvision/OmniSeg3D/assets/37448328/65fc5798-23e0-4b20-b557-c5c23606a6c5)
Towards segmenting everything in 3D all at once, we propose an omniversal 3D segmentation method (a), which takes as input multi-view, inconsistent, class-agnostic 2D segmentations, and then outputs a consistent 3D feature field via a hierarchical contrastive learning framework. This method supports hierarchical segmentation (b), multi-object selection (c), and holistic discretization (d) in an interactive manner.

#### Performance on Replica Room_0
https://github.com/THU-luvision/OmniSeg3D/assets/37448328/f41a256a-e6dd-4f3e-9d59-2089406ac06d

For more demos, please visit our project page: [OmniSeg3D](https://oceanying.github.io/OmniSeg3D/).

## Update
* **2024/01/14**: We release the original version of OmniSeg3D. Try and play with it now!


## Installation

NOTE: Our project is implemented based on the [ngp_pl](https://github.com/kwea123/ngp_pl) project and the requirements are the same as ngp_pl basically.

### Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 8GB (Tested with a single RTX 2080 Ti and RTX 3090), CUDA 11.3 (might work with older version)

### Software

1. Clone this repo by:
	```bash
	git clone https://github.com/THU-luvision/OmniSeg3D.git
 	```
2.  Create a conda environment and activate it, Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended.
	```bash
	conda create -n omniseg3d python=3.8
	conda activate omniseg3d
	```
3. `pytorch`, `pytorch-lightning=1.9.3`, [`pytorch-scatter`](https://github.com/rusty1s/pytorch_scatter#installation)
	```bash
	conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
	conda install pytorch-lightning=1.9.3
	conda install pytorch-scatter -c pyg
	```
4. `tinycudann`: following the official [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension). NOTE: If you want to install it on server with local installed CUDA, you need to specify the CUDA path as `cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.3/bin/nvcc` instead of 'cmake . -B build'.
	```bash
	git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
	cd tiny-cuda-nn/bindings/torch
	python setup.py install
	```
5. `apex`: following the official [instruction](https://github.com/NVIDIA/apex#linux). NOTE: [Error](https://github.com/NVIDIA/apex/issues/1735) may occur due to the recent official commit, try `git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82` to resolve the problem.
	```bash
	git clone https://github.com/NVIDIA/apex
	cd apex
	# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
	# otherwise
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	```
6. `SAM` for segmentation:
	```bash
	git clone https://github.com/facebookresearch/segment-anything.git
	cd segment-anything
 	pip install -e .
	mkdir sam_ckpt; cd sam_ckpt
	wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
	```
7. Other python requirements:
	```bash
	pip install -r requirements.txt
	```

8. Cuda Extension: Upgrade `pip` to >= 22.1 and run:
	```bash
	pip install models/csrc/
 	```

## Data Preparartion
### Hierarchical Representation Generation
Run the sam model to get the hierarchical representation files.
```bash
python run_sam.py --ckpt_path {SAM_CKPT_PATH} --file_path {IMAGE_FOLDER} --gpu_id {GPU_ID}
```
After running, you will get three folders `sam`, `masks`, `patches`:
* `sam`: stores the hierarchical representation as ".npz" files
* `masks` and `patches`: used for visualization or masks quaility evaluation, not needed during training.

Ideal `masks` should include object-level masks and `patches` should contain part-level masks. We basically use the default parameter setting for SAM, but you can tune the parameters for customized datasets.

### Data Sample
We provide some [data sample (replica_room_0, 360_counter, llff_flower)](https://drive.google.com/drive/folders/1e7eCume6solK8NuesWdFe9vabVmA9YYX?usp=sharing), you can download them for model trainning.

### Data structure:  
NOTE: Folder "sam", "masks", and "patches" should be generated with run_sam.py
```
    data
    ├── 360_v2             	# Link: https://jonbarron.info/mipnerf360/
    │   └── [bicycle|bonsai|counter|garden|kitchen|room|stump]
    │       ├── [sparse/0] (colmap results)
    │       └── [images|images_2|images_4|images_8|sam|masks|patches]
    │
    ├── nerf_llff_data     	# Link: https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │       ├── [sparse/0] (colmap results)
    │       └── [images|images_2|images_4|images_8]
    │
    └── replica_data		# Link: https://github.com/ToniRV/NeRF-SLAM/blob/master/scripts/download_replica.bash
        └── [office_0|room_0|...]
            ├── transforms_train.json
            └── [rgb|depth(optional)|sam|masks|patches]
```




## Training

We recommend a two-stage training strategy for stable convergence, which means we train for **color and density field** first and then for **semantic field**. 

* Before running: please specify the information in the config file (like `run_replica.sh`). More options can be found in `opt.py` and you can them adjusted in config file.
```bash
# --- Edit the config file scripts/run_replica.sh
root_dir=/path/to/data/folder/of/the/scene
exp_name=experiment_name
dataset_name=dataset_type  # "colmap", "replica", and you can easily specify new dataset type
```

* Stage 1: color and density field optimization
```bash
CUDA_VISIBLE_DEVICES=0 opt=train_rgb bash scripts/run_replica.sh
```

* Stage2: semantic field optimization
```bash
CUDA_VISIBLE_DEVICES=0 opt=train_sem bash scripts/run_replica.sh
```



## Inference

We provide GUI (based on DearPyGUI) for interactive segmentation.

* Stage1: color and density field visualization
```bash
CUDA_VISIBLE_DEVICES=0 opt=show_rgb bash scripts/run_replica.sh
```

* Stage2: semantic field visualization and segmentation
```bash
CUDA_VISIBLE_DEVICES=0 opt=show_sem bash scripts/run_replica.sh
```

Here are some functional instructions for interactive segmentation in GUI:
* The view point can be changed by dragging the mouse on the screen
* Left click `clickmode` button to start segmentation mode:
   * `Single-click mode`: right click the region of interest, the object or part will be highlighted, and the score map will show the similarity between the selected pixel and other rendered pixels.
   * `Multi-click mode`: choose `multi-clickmode` button, then you can select multiple pixels on the screen by right click them.
   * `Similarity Threshold`: drag the pin of `ScoreThres`, then the unselected regions will be darkened.
   * `Binarization`: left click the `binary threshold` button a binary mask will be applied to the RGB image via the chosen similarity threshold.

#### Trained Models
We provide [trained model for replica room_0](https://drive.google.com/drive/folders/1e7eCume6solK8NuesWdFe9vabVmA9YYX?usp=sharing), you can use it for GUI visulization and interactive segmentation. This sample also reveals the output organization. It is recommended to put the unzipped "results" folder under the root_dir of OmniSeg3D for minimum code modification.


#### Performance on MipNeRF360 Counter
https://github.com/THU-luvision/OmniSeg3D/assets/37448328/29c7b1db-5c34-4c24-b896-9d33b5e66ac3


#### Comparison with [SA3D](https://github.com/Jumpat/SegmentAnythingin3D)
https://github.com/THU-luvision/OmniSeg3D/assets/37448328/99e75832-24b7-4535-9f98-01efad3d33b5




## TODO List
- [ ] Release mesh-based implementation;


## Acknowledgements
Thanks for the following project for their valuable contributions:
- [ngp_pl](https://github.com/kwea123/ngp_pl)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)


## Citation
If you find this project helpful for your research, please consider citing the report and giving a ⭐.
```BibTex
@article{ying2023omniseg3d,
  title={OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning},
  author={Ying, Haiyang and Yin, Yixuan and Zhang, Jinzhi and Wang, Fan and Yu, Tao and Huang, Ruqi and Fang, Lu},
  journal={arXiv preprint arXiv:2311.11666},
  year={2023}
}
```
