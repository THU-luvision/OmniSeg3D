from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset
from .dmnerf import DMNeRFDataset
from .replica import ReplicaDataset
from .replica_small import ReplicaSmallDataset
from .mvsnet import MVSNetDataset
from .pig import PigDataset
from .haoxiang import HaoxiangDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset,
                'dmnerf': DMNeRFDataset,
                'replica': ReplicaDataset,
                'replica_small': ReplicaSmallDataset,
                'mvsnet': MVSNetDataset,
                'pig': PigDataset,
                'haoxiang': HaoxiangDataset,
                }