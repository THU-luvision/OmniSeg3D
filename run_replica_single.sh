
# python train.py \
#     --root_dir /media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/Segment/replica_sam/Replica640/room_0 \
#     --exp_name rgb/1110_room_0 \
#     --dataset_name replica_small \
#     --scale 0.5 \
#     --num_epochs 5 \
#     --batch_size 8192 \
#     --ray_sampling_strategy all_images \
#     --bg_color 0 \
#     --depth_flag \
#     --opacity_loss_w 0.001 \
#     --distortion_loss_w 0.001 \
#     --depth_loss_w 0.1


# python show_gui.py \
#     --root_dir /media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/Segment/replica_sam/Replica640/room_0 \
#     --exp_name rgb/1110_room_0 \
#     --dataset_name replica_small \
#     --scale 0.5 \
#     --bg_color 0 \
#     --depth_flag \
#     --ckpt_path results/replica_small/rgb/1110_room_0/ckpts/epoch=4_slim.ckpt


# python train.py \
#     --root_dir /media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/Segment/replica_sam/Replica640/room_0 \
#     --exp_name sam/1110_room_0 \
#     --dataset_name replica_small \
#     --scale 0.5 \
#     --num_epochs 5 \
#     --batch_size 8192 \
#     --ray_sampling_strategy same_image \
#     --bg_color 0 \
#     --depth_flag \
#     --sam_loss_w 0.0005 \
#     --semantic_flag \
#     --patch_flag \
#     --semantic_dim 16 \
#     --weight_path results/replica_small/rgb/1110_room_0/ckpts/epoch=4.ckpt \
#     --semantic_only


python show_gui_multi_4box.py \
    --root_dir /media/luvision/E8AAE73BAAE704C2/Users/luvision/Downloads/windows-haiyang/Segment/replica_sam/Replica640/room_0 \
    --exp_name sam/1110_room_0 \
    --dataset_name replica_small \
    --scale 0.5 \
    --bg_color 0 \
    --depth_flag \
    --semantic_flag \
    --semantic_dim 16 \
    --ckpt_path results/replica_small/sam/1110_room_0/ckpts/epoch=4_slim.ckpt
