from .utils.models.segmentation_model import SegmentationModel
from .utils.data.augmentation import MultipleRandomRotation, MultipleColorJitter, MultipleToTensor
from .utils.data.dataloader import SegmentationDataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from os.path import join
from tqdm import tqdm
from time import localtime
from torch import optim

import matplotlib.pyplot as plt

from datetime import datetime

# TODO: Add types to args


default_transform = transforms.Compose([MultipleRandomRotation(10, fill=(255, 0)),
                                        MultipleColorJitter(
    brightness=0.3, contrast=0.3, n_max=1),
    MultipleToTensor(),
])


# General base class for Processors.
class BaseProcessor:
    def __init__(self, use_pretrained=True, lock_pretrained=True):
        self.model = SegmentationModel(use_pretrained, lock_pretrained)

    def load_data(self,
                  train_path,
                  val_path=None,
                  transform="default"):
        """
        Loads the data and creates a dataloader.

        Args:
            train_path (str): path to a directory containing two folders named
                `originals` and `labels` for training data.
            val_path (str, optional): path to a directory containing two
                named `originals` and `labels` for validation data.
                Default: None.
            transform (:mod:`~torchvision.transforms`, optional):
                Transforms to be applied on images and labels. It should be 
                able to take a list of images as inputs and return a list of
                transformed images.
        """

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

    def train_step(self,
                   x,
                   y,
                   optimizer,
                   criterion,
                   ):

        optimizer.zero_grad()

        preds = self.model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self,
                 x,
                 y,
                 criterion):

        preds = self.model(x)

        loss = criterion(preds, y)

        return loss.item()

    def train_epoch(self,
                    epoch,
                    train_dl,
                    optimizer,
                    criterion,
                    writer,
                    train_running_loss,
                    log_freq,
                    device,):

        self.model.train()
        total_loss = 0

        # Progress bar
        pbar = tqdm(total=len(train_dl))
        pbar.set_description(f"Epoch: {epoch}. Traininig")

        for i, (x, y) in enumerate(train_dl, 1):

            x = x.to(device)
            y = y.to(device)

            loss = self.train_step(x, y, optimizer, criterion)
            total_loss += loss

            pbar.update()
            pbar.set_postfix(loss=total_loss/(i))

            train_running_loss += loss
            if ((epoch-1)*len(train_dl) + i) % log_freq == 0:

                writer.add_scalar('Loss/train',
                                  train_running_loss/log_freq,
                                  (epoch-1)*len(train_dl) + i)

                pred = self.predict(x[0].unsqueeze(0), device=device)
                _, index = pred.topk(1, dim=1)
                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(index[0, 0].cpu(), "gray")
                ax1.set_title("Prediction")
                ax2.imshow(y[0].cpu(), "gray")
                ax2.set_title("Ground Truth")

                writer.add_figure("Predictions/train",
                                  f,
                                  global_step=(epoch-1)*len(train_dl) + i)

                self.model.train()

                train_running_loss = 0.0

        pbar.close()

        return train_running_loss, total_loss

    def val_epoch(self,
                  epoch,
                  val_dl,
                  criterion,
                  writer,
                  val_running_loss,
                  log_freq,
                  device):
        self.model.eval()
        eval_loss = 0

        # validation progress bar
        pbar = tqdm(total=len(val_dl))
        pbar.set_description(f"Epoch: {epoch}. Validating")

        val_running_loss = 0
        for i, (x, y) in enumerate(val_dl, 1):

            x = x.to(device)
            y = y.to(device)

            loss = self.val_step(x, y, criterion)
            eval_loss += loss

            pbar.update(1)
            pbar.set_postfix(loss=eval_loss/(i))

            val_running_loss += loss
            if ((epoch-1)*len(val_dl) + i) % log_freq == 0:

                writer.add_scalar('Loss/val',
                                  val_running_loss/log_freq,
                                  (epoch-1)*len(val_dl) + i)

                pred = self.predict(x[0].unsqueeze(0), device=device)
                _, index = pred.topk(1, dim=1)
                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(index[0, 0].cpu(), "gray")
                ax1.set_title("Prediction")
                ax2.imshow(y[0].cpu(), "gray")
                ax2.set_title("Ground Truth")

                writer.add_figure("Predictions/val",
                                  f,
                                  global_step=(epoch-1)*len(val_dl) + i)

                val_running_loss = 0.0

        pbar.close()

        return val_running_loss, eval_loss

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              criterion=None,
              optimizer=None,
              weight_decay=0,
              learning_rate=0.0001,
              learning_rate_decay=None,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path=None,
              max_to_keep=5,
              log_dir="./tensorboard_logs",
              log_freq=1000,
              device=None,
              ):

        # check if train loaders are ok
        if train_dl is None:

            # TEST IT

            if getattr(self, "train_dl", None) is None:
                raise Exception(
                    "No DataLoader found. Either pass one in train or use load_data method.")

            train_dl = self.train_dl(
                batch_size, shuffle, num_workers, pin_memory)
        if val_dl is None:
            if getattr(self, "val_dl", None) is None:
                val_dl = None
            else:
                val_dl = self.val_dl(
                    batch_size, shuffle, num_workers, pin_memory)
            ######

        if learning_rate_decay is None:
            learning_rate_decay = 1.0
        self.model.to(device)

        # checkpoint stuff
        if checkpoint_path is None:
            checkpoint_path = join(
                os.getcwd(), "checkpoints", self.__class__.__name__)
        os.makedirs(checkpoint_path, exist_ok=True)
        if checkpoint_freq is None:
            checkpoint_freq = n_epochs//10+1

        current_epoch = 0
        if len(os.listdir(checkpoint_path)) > 0:
            recent_file = self.get_latest_ckpt_path(
                checkpoint_path)

            checkpoint_file_path = join(checkpoint_path, recent_file)
            current_epoch, optimizer, loss = self.load_checkpoint(
                join(checkpoint_file_path), optimizer, device)

            print("Latest checkpoint found.")
            print(
                f"Epoch: {current_epoch}    loss: {loss}\nResuming training...")

        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.param_groups[0]["weight_decay"] = weight_decay
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, learning_rate_decay)

        current_epoch += 1

        now = datetime.now()
        train_running_loss = 0
        if val_dl is not None:
            val_running_loss = 0
        writer = SummaryWriter(join(log_dir, now.strftime("%Y%m%d-%H%M%S")))

        for epoch in range(current_epoch, current_epoch+n_epochs):

            train_running_loss, total_loss = self.train_epoch(
                epoch, train_dl, optimizer, criterion, writer, train_running_loss, log_freq, device)

            if val_dl is not None:
                val_running_loss, eval_loss = self.val_epoch(
                    epoch, val_dl, criterion, writer, val_running_loss, log_freq, device)

            scheduler.step()

            if epoch % checkpoint_freq == 0:
                name = "{}-{}-{}-{}-{}-{}-{}.pt".format(
                    epoch, *localtime()[:6])
                self.save_checkpoint(join(checkpoint_path, name), epoch,
                                     optimizer, total_loss/len(train_dl))

        writer.flush()
        writer.close()

    def save_checkpoint(self, path, epoch, optimizer, loss):
        r"""
        save the checkpoint (epoch, model parameters, optimizer parameters, loss).

        Args:
            path (str): path of the checkpoint file(.pt or .pth).
            epoch (int): epoch number to be saved in the checkpoint.
            optimizer (:class:`~torch.optim.Optimizer`): optimizer whose
                parameters need to be saved.
            loss (float): loss value to be stored.
        """
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, path)
        print("checkpoint saved")

    def load_checkpoint(self, path, optimizer, device):
        r"""
        loads a saved checkpoint.

        Args:
            path (str): path to a .pt or .pth file.
            optimizer (:class:`~torch.optim.Optimizer`): optimizer
                (preferably the same one which was saved) to load it's parameters.
            device (str): device to load on the checkpoint. "cuda" or "cpu".

        Returns:
            tuple(int, :class:`~torch.optim.Optimizer`, float): tuple of epoch,
            optimizer and loss.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)

        optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["epoch"], optimizer, checkpoint["loss"]

    def get_latest_ckpt_path(self, path):
        checkpoints = os.listdir(path)
        recent_time = datetime(
            *list(map(int, checkpoints[0].split(".")[0].split("-")))[1:])
        recent_file = checkpoints[0]
        for filename in checkpoints[1:]:
            date = list(map(int, filename.split(".")[0].split("-")))
            c_time = datetime(*date[1:])
            if c_time > recent_time:
                recent_time = c_time
                recent_file = filename
            elif c_time == recent_time:
                recent_file = filename if date[0] > int(
                    recent_file.split("-")[0]) else recent_file

        return recent_file

    def load_model(self, path):
        """
        Loads the model parameters for :attr:`self.model`.

        Args:
            path (str): path to a .pth(or .pt) file.
        """
        print(self.model.load_state_dict(torch.load(path)))
        print("Model Loaded!")

    def save_model(self, path):
        r"""saves the current model.

        Args:
            path(str): path to the file. (.pt or .pth)
        """
        if not (path.endswith(".pt") or path.endswith(".pth")):
            path = path+".pt"

        torch.save(self.model.state_dict(), path)
        print("Model Saved!")

    def predict(self, x, device=None):
        r"""predicts the output for a given input.

        Args:
            x (:class:`~torch.Tensor`): input tensor of shape 
                ``[batch_size, height, width, 3]``.
            device (str, optional): device on which to perform the computation.
                Default: GPU if it is available, else CPU.
        """
        # TODO: add dataloader option
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        x = x.to(device)
        self.model.eval()
        return self.model(x)


class PageExtractor(BaseProcessor):
    """
    This class is used to detect boundary of text in an image of a document.

    Args:
        use_pretrained (bool, optional): whether to use the parameters of pre-trained
            resnet50 model. 
            Default: ``True``.
        lock_pretrained (bool, optional): whether to lock the layers of pretrained
            model. Only needed if ``use_pretrained`` is ``True``.
            Default: ``True``.
    """

    def load_data(self,
                  train_path,
                  val_path=None,
                  transform="default"):
        """
        Loads the data and creates a dataloader.

        The label images should be:

        - the same name as the their original images (including the extension).
        - the same size as their original images.
        - black background with the page area filled with red color.

        Args:
            train_path (str): path to a directory containing two folders named
                `originals` and `labels` for training data.
            val_path (str, optional): path to a directory containing two
                named `originals` and `labels` for validation data.
                Default: None.
            transform (:mod:`~torchvision.transforms`, optional):
                Transforms to be applied on images and labels. It should be 
                able to take a list of images as inputs and return a list of
                transformed images.
        """
        super().load_data(train_path, val_path, transform)

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              criterion=None,
              optimizer=None,
              weight_decay=0,
              learning_rate=0.0001,
              learning_rate_decay=None,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path=None,
              max_to_keep=5,
              log_dir="./tensorboard_logs",
              log_freq=1000,
              device=None,
              ):
        r"""
        Training method to detect boundary. 

        :attr:`train_dl` and :attr:`val_dl` should be provided if custom dataloaders
        are required. Although, most of the times, :meth:`~PageExtractor.load_data`
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
            num_workers (int, optional): number of processes to load data.
                Default: 4
            pin_memory (bool, optional): whether to pin memory while loading data.
                Default: ``True``.
            criterion (:class:`~torch.nn.module.loss._Loss` or :class:`~torch.nn.module.loss._WeightedLoss`, optional): 
                The loss function to use.
                Default: ``CrossEntropyLoss(reduction="mean")``
            optimizer (:class:`~torch.optim.Optimizer`, optional): The optimizer
                to be used to update the parameters.
                Default: ``Adam(model.parameters(), lr=learning_rate)``
            weight_decay (float, optional): L2 regularization parameter for the optimizer(if it applies).
                Default: 0.
            learning_rate (float, optional): Learning rate to be used for the 
                default optimizer.
                Default: 0.0001.
            learning_rate_decay (float, optional): exponential decay to be used on learning rate.
                Default: None.
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

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction="mean")

        super().train(n_epochs, train_dl, val_dl, batch_size, shuffle,
                      num_workers, pin_memory, criterion, optimizer, weight_decay,
                      learning_rate, learning_rate_decay, save_checkpoints,
                      checkpoint_freq, checkpoint_path, max_to_keep, log_dir,
                      log_freq, device)


class ImageExtractor(BaseProcessor):
    """
    This class is used to detect images in a document.

    Args:
        use_pretrained (bool, optional): whether to use the parameters of pre-trained
            resnet50 model. 
            Default: ``True``.
        lock_pretrained (bool, optional): whether to lock the layers of pretrained
            model. Only needed if ``use_pretrained`` is ``True``.
            Default: ``True``.
    """

    def load_data(self,
                  train_path,
                  val_path=None,
                  transform="default"):
        """
        Loads the data and creates a dataloader.

        The label images should be:

        - the same name as the their original images (including the extension).
        - the same size as their original images.
        - black background with corresponding image area red. 

        Args:
            train_path (str): path to a directory containing two folders named
                `originals` and `labels` for training data.
            val_path (str, optional): path to a directory containing two
                named `originals` and `labels` for validation data.
                Default: None.
            transform (:mod:`~torchvision.transforms`, optional):
                Transforms to be applied on images and labels. It should be 
                able to take a list of images as inputs and return a list of
                transformed images.
        """
        super().load_data(train_path, val_path, transform)

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              criterion=None,
              optimizer=None,
              weight_decay=0,
              learning_rate=0.0001,
              learning_rate_decay=None,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path=None,
              max_to_keep=5,
              log_dir="./tensorboard_logs",
              log_freq=1000,
              device=None,
              ):
        r"""
        Training method to detect images. 


        :attr:`train_dl` and :attr:`val_dl` should be provided if custom dataloaders
        are required. Although, most of the times, :meth:`~ImageExtractor.load_data`
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
            num_workers (int, optional): number of processes to load data.
                Default: 4
            pin_memory (bool, optional): whether to pin memory while loading data.
                Default: ``True``.
            criterion (:class:`~torch.nn.module.loss._Loss` or :class:`~torch.nn.module.loss._WeightedLoss`, optional): 
                The loss function to use.
                Default: ``CrossEntropyLoss(reduction="mean")``
            optimizer (:class:`~torch.optim.Optimizer`, optional): The optimizer
                to be used to update the parameters.
                Default: ``Adam(model.parameters(), lr=learning_rate)``
            weight_decay (float, optional): L2 regularization parameter for the optimizer(if it applies).
                Default: 0.
            learning_rate (float, optional): Learning rate to be used for the 
                default optimizer.
                Default: 0.0001.
            learning_rate_decay (float, optional): exponential decay to be used on learning rate.
                Default: None.            
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

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction="mean")

        super().train(n_epochs, train_dl, val_dl, batch_size, shuffle,
                      num_workers, pin_memory, criterion, optimizer, weight_decay,
                      learning_rate, learning_rate_decay, save_checkpoints,
                      checkpoint_freq, checkpoint_path, max_to_keep, log_dir,
                      log_freq, device)


class BaselineDetector(BaseProcessor):
    """
    This class is used to detect baselines in an image of a document.

    Args:
        use_pretrained (bool, optional): whether to use the parameters of pre-trained
            resnet50 model. 
            Default: ``True``.
        lock_pretrained (bool, optional): whether to lock the layers of pretrained
            model. Only needed if ``use_pretrained`` is ``True``.
            Default: ``True``.
    """

    def load_data(self,
                  train_path,
                  val_path=None,
                  transform="default"):
        """
        Loads the data and creates a dataloader.

        The label images should be:

        - the same name as the their original images.
        - the same size as their original images.
        - black background with red lines of about 5px width representing baselines.


        Args:
            train_path (str): path to a directory containing two folders named
                `originals` and `labels` for training data.
            val_path (str, optional): path to a directory containing two
                named `originals` and `labels` for validation data.
                Default: None.
            transform (:mod:`~torchvision.transforms`, optional):
                Transforms to be applied on images and labels. It should be 
                able to take a list of images as inputs and return a list of
                transformed images.
        """
        super().load_data(train_path, val_path, transform)

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              criterion=None,
              optimizer=None,
              weight_decay=0,
              learning_rate=0.0001,
              learning_rate_decay=None,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path=None,
              max_to_keep=5,
              log_dir="./tensorboard_logs",
              log_freq=1000,
              device=None,
              ):
        r"""
        Training method to detect baselines. 

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
            num_workers (int, optional): number of processes to load data.
                Default: 4
            pin_memory (bool, optional): whether to pin memory while loading data.
                Default: ``True``.
            criterion (:class:`~torch.nn.module.loss._Loss` or :class:`~torch.nn.module.loss._WeightedLoss`, optional): 
                The loss function to use.
                Default: ``CrossEntropyLoss(weight=Tensor[1, 700]), reduction="mean")``
            optimizer (:class:`~torch.optim.Optimizer`, optional): The optimizer
                to be used to update the parameters.
                Default: ``Adam(model.parameters(), lr=learning_rate)``
            weight_decay (float, optional): L2 regularization parameter for the optimizer(if it applies).
                Default: 0.
            learning_rate_decay (float, optional): exponential decay to be used on learning rate.
                Default: None.
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

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(
                weight=torch.Tensor([1, 700]).to(device), reduction="mean")

        super().train(n_epochs, train_dl, val_dl, batch_size, shuffle,
                      num_workers, pin_memory, criterion, optimizer, weight_decay,
                      learning_rate, learning_rate_decay, save_checkpoints,
                      checkpoint_freq, checkpoint_path, max_to_keep, log_dir,
                      log_freq, device)
