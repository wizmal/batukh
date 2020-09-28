import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

MAX_LENGTH = 20


class WordDataset(Dataset):
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


# The model architechture is organized into three modules:
# 1. `ImgEncoder`: A convolutional network.
# -  It will take input of an image of `[3, 60, x]` (channels, height, width)
# and produce the output of `[128, 12, x']`, which is then flattened out
# to `[1536, x']`.

# 2. `Encoder(input_size, hidden_size, n_layers)`:
# - It has a linear layer followed by a GRU layer.
# - It takes a vector of `[1, input_size]` and the linear converts into
# `[1, hidden_size]` vector.
# - Then the `[1, hidden_size]` vector is passed via a GRU with `n_layers`.
# - The hidden input to GRU is initialized as zeros.

# 3. `AttnDecoderRNN(hidden_size, output_size, n_layers)`:
# - It has an embedding layer giving out `hidden_size` length of vectors.
# - Then a GRU layer with `hidden_size` output and `n_layers`.
# - Then a linear layer of `output_size` nodes.
# - Then a softmax.


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)

        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten(end_dim=-2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool(x)
        x = self.flatten(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, device=None):

        super(Encoder, self).__init__()

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_layers = n_layers

        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.em = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)

    def forward(self, input, hidden):

        output = F.relu(self.fc1(input)).view(1, 1, -1)

        embedded = self.em(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size,
                           device=self.device)


class AttnDecoderRNN(nn.Module):

    # TODO: Check if we can dynamically create a linear layer in forward method
    # to replace `self.attn` and eliminate the need for `MAX_LENGTH`
    def __init__(self, hidden_size, output_size, n_layers, dropout_p=0.1,
                 max_length=MAX_LENGTH, device=None):
        super(AttnDecoderRNN, self).__init__()

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device)


class WordDetector:
    def __init__(self,
                 encoder_input=1536,
                 encoder_hidden=128,
                 encoder_nlayers=2,
                 decoder_hidden=128,
                 decoder_nlayers=2,
                 decoder_output=None,
                 image_dir=None,
                 label_path=None,
                 transform=None,
                 device=None):
        """
        params:
        - `encoder_input`: Input size of each vector that goes into the encoder.
           Default: 1536, based on the assumption that the input image is
           `[3, 60, x]`.
        - `encoder_hidden`: hidden size of the GRU layer. Default: 128.
        - `decoder_output`: Size of the output vector. Should be the same as
           the target vector. If `image_dir` and `label_path` is provided, then
           it will be calculated, else, need to supply.
        - `image_dir`: Path to directory containing images, The images should
           be named as "1.png", "2.png", ...
        - `label_path`: Path to the label file. Each label should be separated
           by a newline. Each label should be like: "1: this label".
        """

        if image_dir is None or label_path is None:
            print(
                "No dataloader/dataset made, expect a dataloader in\
                     the `train` method.")
            self.dataset = None
        else:
            self.dataset = WordDataset(image_dir, label_path, transform)
        if decoder_output is None:
            decoder_output = len(self.dataset.letter2index)

        self.model = nn.Sequential(ImgEncoder(),
                                   Encoder(encoder_input, encoder_hidden,
                                           encoder_nlayers, device),
                                   AttnDecoderRNN(decoder_hidden,
                                                  decoder_output,
                                                  decoder_nlayers,
                                                  device=device))

    def train(self, n_epochs):
        pass


# TODO: Test each and every module of this file.
