from argparse import ArgumentParser
import cv2
from glob import glob
import numpy as np
from os import makedirs, remove
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

parser = ArgumentParser()
[parser.add_argument(s) for s in ('--scene', '--in_dir', '--out_dir')]
parser.add_argument('--n_samples', type=int, default=1024)
parser.add_argument('--w_neg_sam', type=float, default=.15)
parser.add_argument('--w_pos_xyz', type=float, default=500000)
parser.add_argument('--w_neg_xyz', type=float, default=500000)
parser.add_argument('--q_pos', type=float, default=.95)
parser.add_argument('--q_neg', type=float, default=1)
args = parser.parse_args()
root = f'data/Replica_Dataset/{args.scene}/Sequence_1'


def load_data(frame: int, index: int) -> list[np.ndarray]:
    image = cv2.imread(f'{root}/rgb/rgb_{frame}.png')
    depth = cv2.imread(f'{root}/depth/depth_{frame}.png', cv2.IMREAD_UNCHANGED)
    label = cv2.imread(f'{root}/semantic_instance/semantic_instance_{frame}.png', cv2.IMREAD_UNCHANGED)
    sam, xyz = np.load(f'{args.in_dir}/sam/{index:03}.npy'), np.load(f'{args.in_dir}/xyz/{index:03}.npy')
    return image, depth.astype(np.int32) ** 2, label.astype(np.uint8), sam, xyz


# Load masks, XYZ coordinates, SAM latents and object IDs
frames = np.arange(900)
if args.scene == 'office_1': frames = np.delete(frames, np.s_[474:504])
elif args.scene == 'office_4': frames = np.delete(frames, np.s_[618:734])
images, depths, labels, sams, xyzs = zip(*thread_map(load_data, frames, range(frames.size), desc='Data'))
torch.set_default_tensor_type(torch.cuda.FloatTensor)
depths, labels, sams, xyzs = [torch.tensor(np.stack(a)) for a in (depths, labels, sams, xyzs)]
sams = torch.nn.functional.normalize(sams, dim=3)
ids = torch.tensor({
    'office_0': [3,4,7,8,9,10,12,14,17,19,21,23,26,28,29,30,36,37,40,42,44,46,54,55,57,58,61],
    'office_1': [3,7,9,11,13,14,15,17,23,24,29,32,33,36,37,39,42,44,45,46],
    'office_2': [2,8,9,13,14,17,19,23,27,40,41,47,49,51,54,58,60,65,67,70,71,72,73,78,85,90,92,93],
    'office_3': [3,8,11,14,15,18,19,25,29,30,32,33,38,39,43,51,54,55,61,65,72,76,78,82,87,91,95,96,101,111],
    'office_4': [1,2,6,7,9,11,17,22,23,26,33,34,39,47,49,51,52,53,55,56],
    'room_0': [5,6,7,10,13,14,16,25,32,33,35,46,51,53,55,60,64,67,68,83,86,87,92],
    'room_1': [1,2,4,6,7,9,10,11,16,18,24,28,32,36,37,44,48,52,54,56],
    'room_2': [3,5,6,7,8,9,11,12,16,18,22,26,27,37,38,39,40,43,49,55,56]
}[args.scene], dtype=torch.uint8)
makedirs(args.out_dir, exist_ok=True)
[remove(file) for file in glob(f'{args.out_dir}/*.png')]

tbar_outer, ious_all = tqdm(enumerate(ids), 'Objects', N_OBJ := ids.numel()), torch.empty(N_OBJ)
for index, label in tbar_outer:

    # Sample positive and negative latents from the reference image
    views = (areas := ((gt := labels == label) * depths).sum((1, 2))).argsort(descending=True)
    ref_view, all_views = views[0], views[:areas.nonzero().numel()]
    W_POS_XYZ, W_NEG_XYZ = [w / areas[views[0]].sqrt() for w in (args.w_pos_xyz, args.w_neg_xyz)]
    sam_pos, sam_neg = sams[ref_view, (pos := labels[ref_view] == label)], sams[ref_view, ~pos]
    stride_pos, stride_neg = 1 + sam_pos.shape[0] // args.n_samples, 1 + sam_neg.shape[0] // args.n_samples
    sam_pos, sam_neg = sam_pos[::stride_pos].T, sam_neg[::stride_neg].T
    xyz_pos, xyz_neg = xyzs[ref_view, pos][::stride_pos], xyzs[ref_view, ~pos][::stride_neg]

    # Determine the best threshold and compute accuracy and IoU
    tbar_inner, probab = tqdm(enumerate(all_views), 'Frames', N_VIEWS := all_views.numel(), False), 1
    threshold, ious = torch.arange(-2 * args.w_neg_sam, 2, 1 / 128).view(-1, 1, 1), torch.empty(N_VIEWS)
    for n, view in tbar_inner:
        similar_pos = (sams[view] @ sam_pos).add_(1).float()
        if W_POS_XYZ > 0: similar_pos *= torch.cdist(xyzs[view], xyz_pos).mul_(-W_POS_XYZ).exp_()
        similar_pos = similar_pos.quantile(args.q_pos, 2) if args.q_pos < 1 else similar_pos.amax(2)
        similar_neg = (sams[view] @ sam_neg).add_(1).float()
        if W_NEG_XYZ > 0: similar_neg *= torch.cdist(xyzs[view], xyz_neg).mul_(-W_NEG_XYZ).exp_()
        similar_neg = similar_neg.quantile(args.q_neg, 2) if args.q_neg < 1 else similar_neg.amax(2)
        pred = similar_pos.add_(similar_neg.mul_(-args.w_neg_sam)) > threshold
        ious[n] = (iou := (pred & gt[view]).sum((-2, -1)) / (pred | gt[view]).sum((-2, -1))).max()
        if n == 0: threshold = threshold[iou.argmax()]
        tbar_inner.set_postfix(iou_frame=ious[n].item(), iou_object=ious[:n + 1].mean().item())

        # Save and visualize the results
        if n == 0 or (ious[n].item() < .75 and np.random.rand() < probab):
            probab *= .5
            (vis := images[view.item()] >> 1)[gt[view].cpu().numpy(), 1] += 128
            vis[(pred if n else pred[iou.argmax()]).cpu().numpy(), 2] += 128
            cv2.imwrite(f'{args.out_dir}/obj{label:03}-{"tgt" if n else "ref"}{view:03}-iou' +
                        f'{round(100 * ious[n].item()):02}.png', vis, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    ious_all[index] = (iou := ious[1:].mean().nan_to_num_(0))
    tbar_outer.set_postfix(iou_object=iou.item(), iou_scene=ious_all[:index + 1].mean().item())
np.savetxt(f'{args.out_dir}.txt', torch.stack((ids, 100 * ious_all), 1).cpu().numpy(), ('%d', '%.2f'))
open(f'{args.out_dir}.txt', 'a').write(f'\nmIoU {100 * ious_all.mean():.2f}\n{args._get_kwargs()[3:]}')
