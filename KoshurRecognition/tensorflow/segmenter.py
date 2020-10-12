from .utils.models.segmentation_model import SegmentationModel
from .utils.train import Train
from .utils.data.dataloader import SegmentationDataLoader
from .utils.data.augmentation import MultipleColorJitter


# todo:apply augmentation


class PageExtraction(Train):
    r"""This class used to extract pages (removing borders and blank spaces around pages) from the images."""

    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  pixels corresponding to page area colored red.

        Args:
            train_path (str)        : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, batch_size=2, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        super().train(n_epochs, train_dl=self.train_dl, val_dl=self.val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)


class ImageExtraction(Train):
    r"""The class used to extract images."""

    def __init__(self):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  pixels corresponding image areas colored red.

        Args:
            train_path (str)        : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, batch_size=2, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        super().train(n_epochs, train_dl=self.train_dl, val_dl=self.val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)


class LayoutExtraction(Train):
    r"""This class is used to extract diffrent layouts from a image."""

    def __init__(self, train_path, val_path, batch_size, repeat,  n_layouts):
        super().__init__(model=SegmentationModel(n_layouts))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  pixels corresponding to diffrent areas colored diffrently.

        Args:
            train_path (str)        : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classes)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, batch_size=2, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        super().train(n_epochs, train_dl=self.train_dl, val_dl=self.val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)


class BaselineDetection(Train):
    r"""This class is used to detect baseline."""

    def __init__(self, train_path, val_path, batch_size, repeat):
        super().__init__(model=SegmentationModel(2))
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.

        Note:
            Respective images and labels should be of same size with same filename.
            label images should be of black background with  red lines of about 5px representing baselines.

        Args:
            train_path (str)        : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for train dataset.
            val_path (str,optional) : path of the folder contaings images folder (containing orginal images) and labels folder (containing label images) for validation dataset.
            """
        self.train_dl = SegmentationDataLoader(
            train_path, self.model.n_classest)
        if val_path is not None:
            self.val_dl = SegmentationDataLoader(
                val_path, self.model.n_classes)

    def train(self, n_epochs, batch_size=2, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        super().train(n_epochs, train_dl=self.train_dl, val_dl=self.val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)
