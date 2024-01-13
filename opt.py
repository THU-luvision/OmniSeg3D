import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv',
                                 'dmnerf', 'replica', 'replica_small', 'mvsnet', 'pig', 'haoxiang'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--aabb_tol', type=float, default=0.1,
                        help='''tolerance for axis-aligned bounding box,
                        points outside of which will have zero density.
                        Used when the actual bounding box is smaller than scale''')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--opacity_loss_w', type=float, default=1e-3,
                        help='weight of opacity loss (see losses.py), 0 to disable')
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--sam_loss_w', type=float, default=1e-4,
                        help='weight of semantic ProtoNCE loss')
    parser.add_argument('--sam_level_w', type=float, default=5,
                        help='inter-level semantic loss down-weighting factor')
    parser.add_argument('--depth_loss_w', type=float, default=1,
                        help='weight of depth loss')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='same_image',
                        choices=['all_images', 'same_image', 'mixture'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    # parser.add_argument('--random_bg', action='store_true', default=False,
    #                     help='''whether to train with random bg color (real scene only)
    #                     to avoid objects with black color to be predicted as transparent
    #                     ''')
    # parser.add_argument('--white_bg', action='store_true', default=False,
    #                     help='whether to train with white bg color (real scene only)')
    parser.add_argument('--bg_color', type=int, default=0,
                        help='background color option, 0: black, 1: white, 2: random')
    parser.add_argument('--semantic_only', action='store_true', default=False,
                        help='whether to load and freeze pretrained density and rgb')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # --- modification by haiyang --- #
    parser.add_argument('--depth_flag', action='store_true', default=False,
                        help='input depth for clustering learning.')
    parser.add_argument('--semantic_flag', action='store_true', default=False,
                        help='input segmentation image for clustering learning.')
    parser.add_argument('--semantic_path', type=str, default='sam',
                        choices=['sam', 'ritm', 'simple'],
                        help='path to segmentation inputs.')
    parser.add_argument('--temporal_validate', action='store_true', default=False,
                        help='validate one image temporarily (to track the segmentation status).')
    parser.add_argument('--semantic_dim', type=int, default=8,
                        help='dimension of semantic feature space')
    parser.add_argument('--patch_flag', action='store_true', default=False,
                        help='input patch segmentation image for clustering learning.')


    return parser.parse_args()
