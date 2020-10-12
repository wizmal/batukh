from .utils.train import Train
from .utils.data.dataloader import OCRDataLoader
from .utils.data.augmentation import MultipleColorJitter
from .utils.models.ocr_model import OCRModel


class OCR(Train):
    r"""This class is used for octical character recognition.

    Args:
        n_classes (int) : number of outputs nodes of ocr model.
    """

    def __init__(self, n_classes):

        super().__init__(model=OCRModel(n_classes), is_ocr=True)
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        r"""Loads Train and Validation datset.
        Args:
            train_path (str)        : path of the folder contaings images folder,labels.txt and table.txt for train dataset.
            val_path (str,optional) : path of the folder contaings images folder ,labels.txt and table.txt  for validation dataset.
            """
        self.train_dl = OCRDataLoader(
            train_path)
        if val_path is not None:
            self.val_dl = OCRDataLoader(
                val_path)

    def train(self, n_epochs, batch_size=2, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        super().train(n_epochs, train_dl=self.train_dl, val_dl=self.val_dl, batch_size=batch_size, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)
