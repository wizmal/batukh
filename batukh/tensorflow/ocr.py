from .utils.train import Train
from .utils.data.dataloader import OCRDataLoader
from .utils.data.augmentation import MultipleColorJitter
from .utils.models.ocr_model import OCRModel
import tensorflow as tf

# todo batch size segmenter.
# todo dynamic  weights.
# todo checkpoints.
# lr sheduler.
# l2 regulirization e-6.

# todo tensorboard
# todo example boun
#


class OCR(Train):
    r"""This class is used for octical character recognition.

    Args:
        n_classes (int) : number of outputs nodes of ocr model.

     Example

    .. code:: python

        >>> from batukh.tensorflow.ocr import OCR
        >>> m = OCR(177)
        >>> m.load_data(train_path="/data/",height=32)
        >>> m.train(1)
    """

    def __init__(self, n_classes):

        super().__init__(model=OCRModel(n_classes), is_ocr=True)
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, height=64):
        r"""Loads Train and Validation datset.

        Args:
            train_path (str)        : path of the folder contaings images folder,labels.txt and table.txt for train dataset.
            val_path (str,optional) : path of the folder contaings images folder ,labels.txt and table.txt  for validation dataset.
            """
        self.train_dl = OCRDataLoader(
            train_path, height)
        if val_path is not None:
            self.val_dl = OCRDataLoader(
                val_path, height)

    def train(self, n_epochs, train_dl=None, val_dl=None, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        if train_dl is None:
            train_dl = self.train_dl
        if val_dl is None:
            val_dl = self.val_dl
        super().train(n_epochs, train_dl=train_dl, val_dl=val_dl, batch_size=1, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)

    def map2string(self, inputs, table=None, blank_index=None):
        if table is None:
            table = self.train_dl.inv_table
        if blank_index is None:
            blank_index = self.train_dl.blank_index
        super().map2string(inputs=inputs, table=table, blank_index=blank_index)

    def decode(self,  inputs, from_pred=True, method='gready', merge_repeated=True, table=None, blank_index=None):
        super().decode(inputs=inputs, from_pred=from_pred,
                       method=method, merge_repeated=merge_repeated, table=table, blank_index=blank_index)
