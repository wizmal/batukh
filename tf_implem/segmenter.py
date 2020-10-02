from utils.models.segementation_model import SegmentationModel
from utils.train import Train
from utils.data.dataloader import SegmentationDataLoader
from utils.data.augmentation import MultipleColorJitter

# todo: modelLoad ,model save,apply augmentation


class PageExtraction(Train):
    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, batch_size=4, repeat=1):
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes, batch_size, repeat)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes, batch_size, repeat)

    def train(self, epochs=10, save_checkpoints=True, checkpoint_freq=5, save_logits=False):
        super().train(train_ds=self.train_dl, val_ds=self.val_dl, epochs=10,
                      save_checkpoints=True, checkpoint_freq=5, save_logits=False)


class ImageExtraction(Train):
    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, batch_size=4, repeat=1):
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes, batch_size, repeat)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes, batch_size, repeat)

    def train(self, epochs=10, save_checkpoints=True, checkpoint_freq=5, save_logits=False):
        super().train(train_ds=self.train_dl, val_ds=self.val_dl, epochs=10,
                      save_checkpoints=True, checkpoint_freq=5, save_logits=False)


class LayoutExtraction(Train):
    def __init__(self, train_path, val_path, batch_size, repeat,  n_layouts):
        super().__init__(model=SegmentationModel(n_layouts))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, batch_size=4, repeat=1):
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes, batch_size, repeat)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes, batch_size, repeat)

    def train(self, epochs=10, save_checkpoints=True, checkpoint_freq=5, save_logits=False):
        super().train(train_ds=self.train_dl, val_ds=self.val_dl, epochs=10,
                      save_checkpoints=True, checkpoint_freq=5, save_logits=False)


class BaselineDetection(Train):
    def __init__(self, train_path, val_path, batch_size, repeat):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, batch_size=4, repeat=1):
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes, batch_size, repeat)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes, batch_size, repeat)

    def train(self, epochs=10, save_checkpoints=True, checkpoint_freq=5, save_logits=False):
        super().train(train_ds=self.train_dl, val_ds=self.val_dl, epochs=10,
                      save_checkpoints=True, checkpoint_freq=5, save_logits=False)
