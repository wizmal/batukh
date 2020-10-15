from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
import torch
from PIL import Image
from torchvision import transforms


class SegmentationDataLoader(Dataset):
    """
    Makes a :class:`~torch.utils.data.Dataset` and a :class:`~torch.utils.data.DataLoader` 
    on call which can be iterated with
    a given batch size.

    Args:
        input_dir (str): path to the directory containing the input images.
        label_dir (str): path to the directory containing the labelled images.
        transforms (transforms or None, optional): transforms that act on a 
            tuple of images and return a tuple of transformed images.
            For some examples, see :class:`~batukh.torch_implem.data.augmentation.MultipleRandomRotation`  
    """

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

    def __call__(self, batch_size=1, shuffle=True, num_workers=4, pin_memory=True):
        """
        Args:
            batch_size (int, optional): size of the each batch for every iteration.
                Default: 1
            shuffle (bool, optional): Whether to shuffle the dataset or not. Default ``True``.

        Returns:
            :class:`~torch.utils.data.DataLoader` : Instance
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)


class OCRDataLoader(Dataset):
    r"""
    Args:
        image_dir (str): Path to the directory containing images with names like `1.png, 2.png`.
        labels_path (str): Path to a file containing labels like "`1. label_1\\n2. label_2`".
        transform (:class:`torchvision.transforms`, optional): A transform 
           (or a composition of multiple transforms) to be applied on the input images.
    """

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

        self.labels = list(
            map(lambda x: x.split(":", 1)[-1].strip(), self.labels))

        self.index2letter = dict(enumerate(set("".join(self.labels)), 2))
        self.index2letter[self.SOS] = "<SOS>"
        self.index2letter[self.EOS] = "<EOS>"
        self.letter2index = {v: k for k, v in self.index2letter.items()}

    def __call__(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.image_dir, self.files[idx]))
        label = self.labels[idx]

        image = self.transform(image)
        label = self.transform_label(label)

        return image, label

    def transform_label(self, label):
        """
        Encodes a label into its vector given the dictionary. Also adds the
        `End of Sentence` value to it.

        Args:
            label (str): A string to be encoded to a vector.

        Returns:
            :class:`~torch.tensor`: Encoded vector.
        """
        label = [self.letter2index[letter] for letter in label]
        label.append(self.EOS)
        return torch.tensor(label, dtype=torch.long).view(-1, 1)
