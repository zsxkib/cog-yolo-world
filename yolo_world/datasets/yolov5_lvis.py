# Copyright (c) Tencent Inc. All rights reserved.
from mmdet.datasets import LVISV1Dataset
from mmyolo.registry import DATASETS

from .utils import RobustBatchShapePolicyDataset


@DATASETS.register_module()
class YOLOv5LVISV1Dataset(RobustBatchShapePolicyDataset, LVISV1Dataset):
    """Dataset for YOLOv5 LVIS Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
