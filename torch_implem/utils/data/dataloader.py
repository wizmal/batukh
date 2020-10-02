from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
import torch
from PIL import Image
from torchvision import transforms


# TODO make it return a complete dataloader, add batch_size as well.


class SegmentationDataLoader(Dataset):

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
        input_image = Image.open(join(
            self.input_dir, self.input_files[idx]))
        # input_image = input_image.resize((1024, 1536))
        label_image = Image.open(join(
            self.label_dir, self.label_files[idx]))
        # label_image = label_image.resize((1024, 1536))

        if self.transforms is not None:
            input_image, label_image = self.transforms(
                (input_image, label_image))

        label_image = (label_image[0, :, :] > (50/255))*1

        return input_image, label_image

    def __call__(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class OCRDataLoader(Dataset):
    def __init__(self, image_dir, labels_path, transform=None):

        if transform is None:
            transform = transforms.Compose([transforms.Grayscale(3),
                                            transforms.ToTensor()])
        self.transform = transform
        self.image_dir = image_dir
        self.files = os.listdir(image_dir)
        self.files.sort(key=lambda x: int(x.split(".")[0]))

        with open(labels_path) as f:
            self.labels = f.readlines()

        # add assertion for checking filenames and label numbers
        assert len(self.files) == len(self.labels)

        print("Building Dictionary. . .")
        self.SOS = 0
        self.EOS = 1

        self.labels = map(lambda x: x.split(":", 1)[-1].strip(), self.labels)

        self.index2letter = dict(enumerate(set("".join(self.labels)), 2))
        self.index2letter[self.SOS] = "<SOS>"
        self.index2letter[self.EOS] = "<EOS>"
        self.letter2index = {v: k for k, v in self.index2letter.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.image_dir, self.files[idx]))
        label = self.labels[idx].split(":", 1)[-1].strip()

        image = self.transform(image)
        label = self.transform_label(label)

        return image, label

    def transform_label(self, label):
        label = [self.letter2index[letter] for letter in label]
        label.append(self.EOS)
        return torch.tensor(label, dtype=torch.long).view(-1, 1)
