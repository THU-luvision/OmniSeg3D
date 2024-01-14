DATASET=replica
ROOTDIR=/data/haiyang/projects/Datasets/data_omniseg3d/release/replica/room_0
EXPNAME=0114_replica_room_0
BGCOLOR=2


if test $opt == "train_rgb"; then
    python train.py \
        --root_dir ${ROOTDIR} \
        --exp_name ${EXPNAME}/rgb \
        --dataset_name ${DATASET} \
        --scale 0.5 \
        --num_epochs 5 \
        --batch_size 8192 \
        --ray_sampling_strategy all_images \
        --bg_color ${BGCOLOR} \
        --depth_flag \
        --opacity_loss_w 0.001 \
        --distortion_loss_w 0.001 \
        --depth_loss_w 1.0


elif test $opt == "show_rgb"; then
    python show_gui.py \
        --root_dir ${ROOTDIR} \
        --exp_name ${EXPNAME}/rgb \
        --dataset_name ${DATASET} \
        --scale 0.5 \
        --bg_color ${BGCOLOR} \
        --depth_flag \
        --ckpt_path results/${DATASET}/${EXPNAME}/rgb/ckpts/epoch=4_slim.ckpt


elif test $opt == "train_sem"; then
    python train.py \
        --root_dir ${ROOTDIR} \
        --exp_name ${EXPNAME}/sem \
        --dataset_name ${DATASET} \
        --scale 0.5 \
        --num_epochs 5 \
        --batch_size 8192 \
        --ray_sampling_strategy same_image \
        --bg_color ${BGCOLOR} \
        --depth_flag \
        --sam_loss_w 0.0005 \
        --semantic_flag \
        --patch_flag \
        --semantic_dim 16 \
        --weight_path results/${DATASET}/${EXPNAME}/rgb/ckpts/epoch=4.ckpt \
        --semantic_only


elif test $opt == "show_sem"; then
    python show_gui_multi_4box.py \
        --root_dir ${ROOTDIR} \
        --exp_name ${EXPNAME}/sem \
        --dataset_name ${DATASET} \
        --scale 0.5 \
        --downsample 0.5 \
        --bg_color ${BGCOLOR} \
        --depth_flag \
        --semantic_flag \
        --semantic_dim 16 \
        --ckpt_path results/${DATASET}/${EXPNAME}/sem/ckpts/epoch=4_slim.ckpt

fi
