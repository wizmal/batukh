from utils.models.segmentation_model import SegmentationModel
from utils.data.augmentation import MultipleRandomRotation, MultipleColorJitter, MultipleToTensor
from utils.data.dataloader import SegmentationDataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from os.path import join
from tqdm import tqdm
from time import localtime
from torch import optim


transform = transforms.Compose([MultipleRandomRotation(10, fill=(255, 0)),
                                MultipleColorJitter(
                                    brightness=0.3, contrast=0.3, n_max=1),
                                MultipleToTensor(),
                                ])


# General base class for Processors.
class BaseProcessor:
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
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(
                weight=torch.Tensor([1, 700]).to(device), reduction="mean")

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

        if criterion is None:
            criterion = nn.CrossEntropyLoss(
                weight=torch.Tensor([1, 700]).to(device), reduction="mean")

        x = x.to(device)
        y = y.to(device)

        preds = self.model(x)

        loss = criterion(preds, y)

        return loss.item()

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              optimizer=None,
              criterion=None,
              learning_rate=0.0001,
              batch_size=1,
              shuffle=True,
              device=None,
              checkpoint_path="./",
              save_every=None):

        checkpoint_path = join(checkpoint_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        if save_every is None:
            save_every = n_epochs//10+1

        for epoch in range(n_epochs):
            if train_dl is None:
                if self.train_dl is None:
                    raise Exception("No training loaders found.")
                else:
                    train_dl = self.train_dl(batch_size, shuffle)
            if val_dl is None:
                val_dl = self.val_dl

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
            if epoch % save_every == 0:
                self.save_model(checkpoint_path, epoch)

    def predict(self, x):

        self.model.eval()
        return self.model(x)


class BaselineDetector(BaseProcessor):

    def __init__(self,
                 transforms=transform,
                 train_dir=None,
                 val_dir=None):
        """
        `transforms`: `torchvision.transforms` to apply to the dataset.
                       Set `None` for no transform.

        `train_dir`: Path to the root directory of the train dataset.
                     It should contain two subdirectories,
                     named: "originals" and "labels".
                     If `None`, then you will have to pass a `DataLoader` \
                         object in the `train` function.

        `val_dir`: Path to the root directory of the validation dataset.
                   It should contain two subdirectories, named:\
                        "originals" and "labels".

        `batch_size`: size of a batch in data loaders(both train and val).
                      Default:2;
                      Only works if `train_dir` or `val_dir` is not `None`.

        `shuffle`: bool: whether to shuffle both datasets or not.
                   Only works if `train_dir` or `val_dir` is not `None`.
        """
        self.model = SegmentationModel()
        self.transform = transform

        self.train_dl = self.make_data(train_dir)

        self.val_dl = self.make_data(val_dir)

    def make_data(self,
                  directory):
        if directory is not None:
            if "originals" in os.listdir(directory) and\
                    "labels" in os.listdir(directory):
                dataloader = SegmentationDataLoader(
                    join(directory, "originals"),
                    join(directory, "labels"),
                    transforms=self.transform
                )
            else:
                raise Exception(f"The path: {directory} does not contain\
                    'originals' or 'labels' directories.")

            return dataloader
        return None

    def save_checkpoint(self, checkpoint, path):
        """To be used instead of `save_model` later on!"""
        # TODO: code here
        pass

    def load_model(self, path):
        print(self.model.load_state_dict(torch.load(path)))
        print("Model Loaded!")

    def save_model(self, path, postfix=0):
        name = "{} {}-{}-{} {}.{}.{}.pt".format(postfix, *localtime()[:6])
        torch.save(self.model.state_dict(), join(path, name))
        print("Model Saved!")
