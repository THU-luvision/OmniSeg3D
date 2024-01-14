from .colmap import ColmapDataset
from .replica import ReplicaDataset
from .replica_small import ReplicaSmallDataset
from .mvsnet import MVSNetDataset


dataset_dict = {
    'colmap': ColmapDataset,
    'replica': ReplicaDataset,
    'replica_small': ReplicaSmallDataset,
    'mvsnet': MVSNetDataset,
}