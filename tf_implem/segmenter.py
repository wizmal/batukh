from segementation_model import SegmentationModel
from train import Train
from dataloader import DataLoader, Augmentation


class PageExtraction(Train):
    def __init__(self):
        super().__init__(model=SegmentationModel(2))


class ImageExtraction(Train):
    def __init__(self):
        super().__init__(model=SegmentationModel(2))


class LayoutExtraction(Train):
    def __init__(self, n_layouts):
        super().__init__(model=SegmentationModel(n_layouts))


class BaselineDetection(Train):
    def __init__(self):
        super().__init__(model=SegmentationModel(2))
