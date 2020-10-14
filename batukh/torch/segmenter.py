from .utils.models.segmentation_model import SegmentationModel
from .utils.data.augmentation import MultipleRandomRotation, MultipleColorJitter, MultipleToTensor
from .utils.data.dataloader import SegmentationDataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from os.path import join
from tqdm import tqdm
from time import localtime
from torch import optim


default_transform = transforms.Compose([MultipleRandomRotation(10, fill=(255, 0)),
                                        MultipleColorJitter(
    brightness=0.3, contrast=0.3, n_max=1),
    MultipleToTensor(),
])


# General base class for Processors.
class BaseProcessor:
    def __init__(self):
        self.model = SegmentationModel()

    def load_data(self,
                  train_path,
                  val_path=None,
                  transform="default"):

        if transform == "default":
            transform = default_transform

        self.train_dl = self._make_data(train_path, transform)
        self.val_dl = self._make_data(val_path, transform)

    def _make_data(self,
                   directory,
                   transform):
        if directory is not None:
            if "originals" in os.listdir(directory) and\
                    "labels" in os.listdir(directory):
                dataloader = SegmentationDataLoader(
                    join(directory, "originals"),
                    join(directory, "labels"),
                    transforms=transform
                )
            else:
                raise Exception(f"The path: {directory} does not contain\
                    'originals' or 'labels' directories.")

            return dataloader
        return None


# TODO: move self.model.to(device) from `train_step` and `val_step` to `train`


    def train_step(self,
                   x,
                   y,
                   optimizer=None,
                   criterion=None,
                   device=None,
                   learning_rate=0.0001):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        preds = self.model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self,
                 x,
                 y,
                 criterion=None,
                 device=None):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        x = x.to(device)
        y = y.to(device)

        preds = self.model(x)

        loss = criterion(preds, y)

        return loss.item()

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              criterion=None,
              optimizer=None,
              learning_rate=0.0001,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path="./",  # Change to None
              max_to_keep=5,
              device=None,
              ):

        checkpoint_path = join(checkpoint_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        if checkpoint_freq is None:
            checkpoint_freq = n_epochs//10+1

        for epoch in range(n_epochs):
            if train_dl is None:

                # TEST IT

                if getattr(self, "train_dl", None) is None:
                    raise Exception(
                        "No DataLoader found. Either pass one in train or use load_data method.")

                train_dl = self.train_dl(batch_size, shuffle)
            if val_dl is None:
                if getattr(self, "val_dl", None) is None:
                    val_dl = None
                else:
                    val_dl = self.val_dl(batch_size, shuffle)
                ######
            self.model.train()
            total_loss = 0

            # Progress bar
            pbar = tqdm(total=len(train_dl))
            pbar.set_description(f"Epoch: {epoch}. Traininig")

            for i, (x, y) in enumerate(train_dl):
                loss = self.train_step(x, y, optimizer, criterion,
                                       learning_rate=learning_rate, device=device)
                total_loss += loss

                pbar.update()
                pbar.set_postfix(loss=total_loss/(i+1))
            pbar.close()

            if val_dl is not None:
                self.model.eval()
                eval_loss = 0

                # validation progress bar
                pbar = tqdm(total=len(val_dl))
                pbar.set_description(f"Epoch: {epoch}. Validating")

                for i, (x, y) in enumerate(val_dl):
                    loss = self.val_step(x, y, criterion, device)
                    eval_loss += loss

                    pbar.update(1)
                    pbar.set_postfix(loss=eval_loss/(i+1))
                pbar.close()
            if epoch % checkpoint_freq == 0:
                self.save_model(checkpoint_path, epoch)

    def save_checkpoint(self, checkpoint, path):
        # To be used instead of `save_model` later on!
        # TODO: code here
        pass

    def load_model(self, path):
        """
        Loads the model parameters for ``self.model``.

        Args:
            path (str): path to a .pth(or .pt) file.
        """
        print(self.model.load_state_dict(torch.load(path)))
        print("Model Loaded!")

    def save_model(self, path, postfix=0):
        name = "{} {}-{}-{} {}.{}.{}.pt".format(postfix, *localtime()[:6])
        torch.save(self.model.state_dict(), join(path, name))
        print("Model Saved!")

    def predict(self, x):
        # TODO: add dataloader option

        self.model.eval()
        return self.model(x)

# TODO: Add types to args


class BaselineDetector(BaseProcessor):
    """
    This class is used to detect baselines in an image of a document.
    """

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              criterion=None,
              optimizer=None,
              learning_rate=0.0001,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path="./",
              max_to_keep=5,
              device=None):
        r"""
        Training method to detect baselines. 

        The label images should be:

        - the same name as the their original images.
        - the same size as their original images.
        - black background with red lines of about 5px width representing baselines.

        :attr:`train_dl` and :attr:`val_dl` should be provided if custom dataloaders
        are required. Although, most of the times, :meth:`~BaselineDetector.load_data`
        will do the job. 

        Note:

            If :attr:`train_dl` and/or :attr:`val_dl` are provided, then any already made
            dataloader using  :meth:`~BaselineDetector.load_data` will not be used.

        Args:
            n_epochs (int): number of epochs.
            train_dl (:class:`~torch.utils.data.DataLoader`, optional): data loader to train on.
                Default: None.
            val_dl (:class:`~torch.utils.data.DataLoader`, optional): data loader to validate on.
                Default: None.
            batch_size (int, optional): batch size of the data loader created by :meth:`~BaselineDetector.load_data`.
                Default: 1.
            shuffle (bool, optional): whether to shuffle the data loader 
                created by :meth:`~BaselineDetector.load_data`.
            criterion (:class:`~torch.nn.module.loss._Loss` or :class:`~torch.nn.module.loss._WeightedLoss`, optional): 
                The loss function to use.
                Default: ``CrossEntropyLoss(weight=Tensor[1, 700]), reduction="mean")``
            optimizer (:class:`~torch.optim.Optimizer`, optional): The optimizer
                to be used to update the parameters.
                Default: ``Adam(model.parameters(), lr=learning_rate)``
            learning_rate (float, optional): Learning rate to be used for the 
                default optimizer.
                Default: 0.0001.
            save_checkpoints (bool, optional): Whether to save training checkpoints.
                Default: ``True``.
            checkpoint_freq (int, optional): The saving frequency. After each 
                these many epochs, a checkpoint will be saved.
                Default: :math:`\left \lfloor\frac{\text{n_epochs}}{10}\right \rfloor + 1`.
            checkpoint_path (str, optional): Path of the checkpoint folder.
                Default: "./"
            max_to_keep (int or None, optional): The maximum number of latest 
                checkpoints to be saved. ``None`` to save all. 
                Default: 5.
            device (str, optional): The device to do computations on.
                Default: GPU, if GPU is available, else CPU. 
        """

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(
                weight=torch.Tensor([1, 700]).to(device), reduction="mean")
        super().train(n_epochs, train_dl, val_dl, batch_size, shuffle, criterion, optimizer,
                      learning_rate, save_checkpoints, checkpoint_freq, checkpoint_path, max_to_keep, device)
