from torchvision.models.resnet import ResNet, Bottleneck
from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

# Data Prep

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np


class MyDataset(Dataset):

    def __init__(self, input_dir, label_dir, transforms=None):

        self.transforms = transforms
        self.input_dir = input_dir
        self.label_dir = label_dir

        self.input_files = sorted(os.listdir(input_dir))
        self.label_files = sorted(os.listdir(label_dir))

        assert self.input_files == self.label_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_image = Image.open(os.path.join(
            self.input_dir, self.input_files[idx]))
        input_image = input_image.resize((1024, 1536))
        label_image = Image.open(os.path.join(
            self.label_dir, self.label_files[idx]))
        label_image = label_image.resize((1024, 1536))

        if self.transforms is not None:
            input_image, label_image = self.transforms(
                (input_image, label_image))

        label_image = (label_image[0, :, :] > (50/255))*1

        return input_image, label_image


# Transforms


class MultipleRandomRotation(transforms.RandomRotation):
    def __init__(self,
                 degrees,
                 resample=False,
                 expand=False,
                 center=None,
                 fill=None):
        super(MultipleRandomRotation, self).__init__(
            degrees, resample, expand, center, fill)

    def __call__(self, images):

        if self.fill is None:
            self.fill = [None]*len(images)

        angle = self.get_params(self.degrees)
        return [TF.rotate(img, angle, self.resample, self.expand, self.center,
                          self.fill[i]) for i, img in enumerate(images)]


class MultipleColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, n_max=None):
        super(MultipleColorJitter, self).__init__(
            brightness, contrast, saturation, hue)

        self.n_max = n_max

    def __call__(self, images):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        if self.n_max is None:
            self.n_max = len(images)

        out = [transform(images[i]) for i in range(self.n_max)]
        out.extend(images[self.n_max:])
        return out


class MultipleToTensor(transforms.ToTensor):
    def __init__(self):
        super(MultipleToTensor, self).__init__()

    def __call__(self, images):
        return [TF.to_tensor(img) for img in images]


# Models


class Identity(nn.Module):
    """just to delete some layers in pretrained models"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ModifiedBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):

        super(ModifiedBottleneck, self).__init__(inplanes, planes, stride, downsample, groups,
                                                 base_width, dilation, norm_layer)

        self.is_last = is_last

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.is_last:
            out += identity
        out = self.relu(out)

        return out


class ResNetLayer(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, sec_output_block=None, strides=None, connect_last=False):
        super(ResNetLayer, self).__init__()

        if strides is None:
            strides = [1]*blocks

        self.blocks = blocks
        self.sec_output_block = sec_output_block
        planes = int(planes/block.expansion)

        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(planes * block.expansion, eps=1e-05,
                           momentum=0.1, affine=True, track_running_stats=True)
        )

        self.add_module("0", block(inplanes, planes, strides[0], downsample))
        inplanes = int(planes * block.expansion)

        for i in range(1, blocks-1):
            self.add_module(str(i), block(inplanes, planes, strides[i]))
        i += 1
        if connect_last:
            is_last = False
        else:
            is_last = True
        self.add_module(str(i), block(
            inplanes, planes, strides[i], is_last=is_last))

    def forward(self, x):

        for i in range(self.blocks):
            block = getattr(self, str(i))
            x = block(x)
            if i == self.sec_output_block:
                sec_output = x.clone()
        if self.sec_output_block is not None:
            return x, sec_output
        return x


class MyResNet(nn.Module):
    def __init__(self, block, layers, strides=None, norm_layer=None, sec_output_blocks=None):
        super(MyResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if strides is None:
            strides[None]*len(layers)
        if sec_output_blocks is None:
            sec_output_blocks = [None]*layers

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion, layers[0],
                                  strides=strides[0], sec_output_block=sec_output_blocks[0])

        self.inplanes = self.inplanes*block.expansion

        self.layer2 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion//2, layers[1],
                                  strides=strides[1], sec_output_block=sec_output_blocks[1])

        self.inplanes = self.inplanes*block.expansion//2

        self.layer3 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion//2, layers[2],
                                  strides=strides[2], sec_output_block=sec_output_blocks[2])

        self.inplanes = self.inplanes*block.expansion//2

        self.layer4 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion//2, layers[3],
                                  strides=strides[3], sec_output_block=sec_output_blocks[3])

    def forward(self, x):
        sec_outputs = []

        sec_outputs.append(x.clone())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        sec_outputs.append(x.clone())

        x = self.maxpool(x)

        x, sec = self.layer1(x)
        sec_outputs.append(sec)

        x, sec = self.layer2(x)
        sec_outputs.append(sec)

        x, sec = self.layer3(x)
        sec_outputs.append(sec)

        x = self.layer4(x)

        return x, sec_outputs


class UpScalerUnit(nn.Module):
    def __init__(self, n_current, n_output, n_copy):
        super(UpScalerUnit, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n_current+n_copy, n_output,
                              kernel_size=(3, 3), padding=1, bias=False)

    def forward(self, x, copy):

        x = self.upsample(x)
        x = torch.cat((copy, x), dim=1)
        x = self.conv(x)
        return x


class UpScaler(nn.Module):
    def __init__(self, n_classes):
        super(UpScaler, self).__init__()

        self.upscale1 = UpScalerUnit(512, 512, 512)

        self.upscale2 = UpScalerUnit(512, 256, 512)

        self.upscale3 = UpScalerUnit(256, 128, 256)

        self.upscale4 = UpScalerUnit(128, 64, 64)

        self.upscale5 = UpScalerUnit(64, 32, 3)

        self.conv2 = nn.Conv2d(32, n_classes, kernel_size=1, bias=False)

    def forward(self, x, copies):

        x = self.upscale1(x, copies.pop(-1))
        x = self.upscale2(x, copies.pop(-1))
        x = self.upscale3(x, copies.pop(-1))
        x = self.upscale4(x, copies.pop(-1))
        x = self.upscale5(x, copies.pop(-1))

        x = self.conv2(x)

        return x


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.downsampler = MyResNet(ModifiedBottleneck, [3, 4, 6, 3],
                                    strides=[[1, 1, 2], [1, 1, 1, 2],
                                             [1, 1, 1, 1, 1, 2], None],
                                    sec_output_blocks=[1, 2, 4, None])

        resnet = models.resnet50(pretrained=True)
        resnet.avgpool = Identity()
        resnet.fc = Identity()

        print(self.downsampler.load_state_dict(resnet.state_dict()))
        for param in self.downsampler.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=(1, 1), bias=False)
        self.upscaler = UpScaler(2)

    def forward(self, x):
        x, sec_outputs = self.downsampler(x)
        sec_outputs[-1] = self.conv1(sec_outputs[-1])
        x = self.conv2(x)
        x = self.upscaler(x, sec_outputs)

        return x


transform = transforms.Compose([MultipleRandomRotation(10, fill=(255, 0)),
                                MultipleColorJitter(
                                    brightness=0.3, contrast=0.3, n_max=1),
                                MultipleToTensor(),
                                ])


class BaselineDetector:

    def __init__(self,
                 transforms=transform,
                 train_dir=None,
                 val_dir=None,
                 batch_size=2,
                 shuffle=True):
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
        self.model = BaseModel()
        self.transform = transform

        self.train_dataset, self.train_dl = self.make_data(
            train_dir, batch_size, shuffle)

        self.val_dataset, self.val_dl = self.make_data(
            val_dir, batch_size, shuffle)

    def make_data(self,
                  directory,
                  batch_size,
                  shuffle):
        if directory is not None:
            if "originals" in os.listdir(directory) and\
                    "labels" in os.listdir(directory):
                dataset = MyDataset(
                    os.path.join(directory, "originals"),
                    os.path.join(directory, "labels"),
                    transforms=self.transform
                )
                dataloader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle)
            else:
                raise Exception(f"The path: {directory} does not contain\
                    'originals' or 'labels' directories.")

            return dataset, dataloader
        return None, None

    def train_step(self,
                   x,
                   y,
                   optimizer=None,
                   criterion=None,
                   device=None,
                   learning_rate=0.01):

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
              device=None):

        for epoch in range(n_epochs):
            if train_dl is None:
                if self.train_dl is None:
                    raise Exception("No training loaders found.")
                else:
                    train_dl = self.train_dl
            if val_dl is None:
                val_dl = self.val_dl

            self.model.train()
            total_loss = 0

            # Progress bar
            pbar = tqdm(total=len(train_dl))
            pbar.set_description(f"Epoch: {epoch}. Traininig")

            for x, y in train_dl:
                loss = self.train_step(x, y, optimizer, criterion, device)
                total_loss += loss

                pbar.update()
                pbar.set_postfix(loss=total_loss)
            pbar.close()

            if val_dl is not None:
                self.model.eval()
                eval_loss = 0

                # validation progress bar
                pbar = tqdm(total=len(val_dl))
                pbar.set_description(f"Epoch: {epoch}. Validating")

                for x, y in val_dl:
                    loss = self.val_step(x, y, criterion, device)
                    eval_loss += loss

                    pbar.update(1)
                    pbar.set_postfix(loss=eval_loss)
                pbar.close()
            self.save_checkpoint(checkpoint=None, path=None)

    def predict(self, x):

        self.model.eval()
        return self.model(x)

    # finish this later
    def save_checkpoint(self, checkpoint, path):
        pass

    def load_model(self, path):
        print(self.model.load_state_dict(path))
        print("Model Loaded!")

    def save_model(self, path):
        print(torch.save(self.model.state_dict(), path))
        print("Model Saved!")
