import torch
from torch import nn
import torch.nn.functional as F


# The model architechture is organized into three modules:
# 1. `ImgEncoder`: A convolutional network.
# -  It will take input of an image of `[3, 60, x]` (channels, height, width)
# and produce the output of `[128, 12, x']`, which is then flattened out
# to `[1536, x']`. Minimum height of image, x = 16.

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

        # self.em = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)

    def forward(self, input, hidden):

        output = F.relu(self.fc1(input)).view(1, 1, -1)

        # embedded = self.em(input).view(1, 1, -1)
        # output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size,
                           device=self.device)


class AttnDecoderRNN(nn.Module):

    # TODO: Check if we can dynamically create a linear layer in forward method
    # to replace `self.attn` and eliminate the need for `MAX_LENGTH`
    def __init__(self, hidden_size, output_size, n_layers, dropout_p,
                 max_length, device=None):
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

        if self.output_size is not None:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        if self.output_size is not None:
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

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device)
