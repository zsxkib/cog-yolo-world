# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Sequence

import torch
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.logging import print_log
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset


class RobustBatchShapePolicyDataset(BatchShapePolicyDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def _prepare_data(self, idx: int) -> Any:
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

    def prepare_data(self, idx: int, timeout=10) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        try:
            return self._prepare_data(idx)
        except Exception as e:
            if timeout <= 0:
                raise e
            print_log(f'Failed to prepare data, due to {e}.'
                      f'Retrying {timeout} attempts.')
            if not self.test_mode:
                idx = random.randrange(len(self))
            return self.prepare_data(idx, timeout=timeout - 1)


@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to_tensor(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)

    if hasattr(data_batch[0]['data_samples'], 'texts'):
        batch_texts = [meta['data_samples'].texts for meta in data_batch]
        collated_results['data_samples']['texts'] = batch_texts

    if hasattr(data_batch[0]['data_samples'], 'is_detection'):
        # detection flag
        batch_detection = [meta['data_samples'].is_detection
                           for meta in data_batch]
        collated_results['data_samples']['is_detection'] = torch.tensor(
            batch_detection)

    return collated_results
