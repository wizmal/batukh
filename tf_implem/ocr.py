from utils.train import Train
from utils.data.dataloader import OCRDataLoader
from utils.data.augmentation import MultipleColorJitter
from utils.models.ocr_model import OCRModel


class OCR(Train):
    def __init__(self, n_classes):
        super().__init__(model=OCRModel(n_classes), is_ocr=True)
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None):
        self.train_dl = OCRDataLoader(
            train_path)
        if val_path is not None:
            self.val_dl = OCRDataLoader(
                val_path)

    def train(self, epochs=10, batch_size=8, repeat=1, save_checkpoints=True, checkpoint_freq=5, save_logits=False):
        super().train(train_ds=self.train_dl, val_ds=self.val_dl, epochs=10, batch_size=64, repeat=1,
                      save_checkpoints=True, checkpoint_freq=5, save_logits=False)
