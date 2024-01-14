
root_dir=/data/haiyang/projects/Datasets/data_omniseg3d/nerf_llff_data/orchids
exp_name=0113_orchids
dataset_name=colmap
bg_color=2
scale=2.0
downsample=0.25
num_epoch_rgb=10
num_epoch_sem=10
batch_size_rgb=8192
batch_size_sem=8192

if test $opt == "train_rgb"; then
    python train.py \
        --root_dir ${root_dir} \
        --exp_name ${exp_name}/rgb \
        --dataset_name ${dataset_name} \
        --scale ${scale} \
        --downsample ${downsample} \
        --num_epochs ${num_epoch_rgb} \
        --batch_size ${batch_size_rgb} \
        --ray_sampling_strategy all_images \
        --bg_color ${bg_color} \
        --opacity_loss_w 0.001 \
        --distortion_loss_w 0.001

elif test $opt == "show_rgb"; then
    python show_gui.py \
        --root_dir ${root_dir} \
        --exp_name ${exp_name}/rgb \
        --dataset_name ${dataset_name} \
        --scale ${scale} \
        --downsample ${downsample} \
        --bg_color ${bg_color} \
        --ckpt_path results/${dataset_name}/${exp_name}/rgb/ckpts/epoch=$(expr ${num_epoch_rgb} - 1)_slim.ckpt


elif test $opt == "train_sem"; then
    python train.py \
        --root_dir ${root_dir} \
        --exp_name ${exp_name}/sem \
        --dataset_name ${dataset_name} \
        --split train \
        --scale ${scale} \
        --downsample ${downsample} \
        --num_epochs ${num_epoch_sem} \
        --batch_size ${batch_size_sem} \
        --ray_sampling_strategy same_image \
        --bg_color ${bg_color} \
        --sam_loss_w 0.0005 \
        --sam_level_w 2 \
        --patch_flag \
        --semantic_flag \
        --semantic_dim 16 \
        --semantic_only \
        --weight_path results/${dataset_name}/${exp_name}/rgb/ckpts/epoch=$(expr ${num_epoch_rgb} - 1)_slim.ckpt



# --- semantic visualization
#    - opt1: show_gui_multi_4box.py: For locally usage (recommand), to have better visualization effect
#    - opt2: show_gui_sem.py: For remote/server usage, to have lower streamming delay

elif test $opt == "show_sem"; then
    python show_gui_sem.py \
        --root_dir ${root_dir} \
        --exp_name ${exp_name}/sem \
        --dataset_name ${dataset_name} \
        --scale ${scale} \
        --downsample ${downsample} \
        --bg_color ${bg_color} \
        --depth_flag \
        --semantic_flag \
        --semantic_dim 16 \
        --ckpt_path results/${dataset_name}/${exp_name}/sem/ckpts/epoch=$(expr ${num_epoch_rgb} - 1)_slim.ckpt

fi
