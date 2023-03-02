# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CSRDataset(CityscapesDataset):

    CLASSES = ('background', 'road')
    #PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)
