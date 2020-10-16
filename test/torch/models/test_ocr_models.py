import pytest
from batukh.torch.utils.models.ocr_model import ImgEncoder, Encoder, AttnDecoderRNN
import torch
import random
import numpy as np


@pytest.mark.parametrize("batch_size, im_length", [
    (1, 16), (1, 16), (1, 400), (3, 16), (3, 20), (16, 60)] +
    list(zip(np.random.randint(1, 16, 20).tolist(), np.random.randint(16, 20, 20).tolist())))
def test_imgencoder_out_shape(batch_size, im_length):
    model = ImgEncoder()
    inp = torch.rand([batch_size, 3, 60, im_length])
    out = model(inp)

    assert all(out.shape == np.array(
        [batch_size, 1536, (((im_length-4)//2)-4)//2]))


@pytest.mark.parametrize("input_size, hidden_size, n_layers",
                         [(1536, 128, 2),  # default setting
                          (1, 1, 1),
                          (1, 4, 1)] +
                         list(zip(np.random.randint(1, 2000, 200),
                                  np.random.randint(1, 2000, 200),
                                  np.random.randint(1, 5, 200)))
                         )
def test_encoder_out_shape(input_size, hidden_size, n_layers):
    model = Encoder(input_size, hidden_size, n_layers, device="cpu")
    inp = torch.rand([1, input_size])
    hidden = model.initHidden()
    out, hidden = model(inp, hidden)

    assert all(out.shape == np.array([1, 1, hidden_size])) and all(
        hidden.shape == np.array([n_layers, 1, hidden_size]))
