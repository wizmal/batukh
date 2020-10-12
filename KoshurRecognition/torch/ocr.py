import torch
from torch import nn, optim
from torchvision import transforms
import os
import random

from .utils.data.dataloader import OCRDataLoader

from .utils.models.ocr_model import ImgEncoder, Encoder, AttnDecoderRNN

from os.path import join
from tqdm import tqdm
from time import localtime

MAX_LENGTH = 50


class WordDetector:
    r"""
    A basic CRNN which detects the characters in an image.

    Note:
        The :attr:`encoder_input` has been set to 1536 as default. This has
        been done by keeping in mind that the height of each image is 60px. If
        your images have a different height, say, :math:`h`, then encoder_input
        should be 
        :math:`\left \lfloor \frac{\left \lfloor \frac{h-4}{2} \right \rfloor -4}{2} \right \rfloor \times 128`.

    Args:
        encoder_input (int, optional): Input size of vector that goes into the encoder.
            Default: 1536, based on the assumption that the input shape is
            `[batch_size, 3, 60, x]`.
        encoder_hidden (int, optional): hidden size of the GRU layer of encoder. 
            Default: 128.
        encoder_nlayers (int, optional): number of layers in the GRU layer of encoder.
            Default: 2.
        decoder_hidden (int, optional): hidden size of the GRU layer of decoder.
            Default: 128.
        decoder_nlayers (int, optional): number of layers in the GRU layer of decoder.
            Default: 2.
        decoder_output (int, optional): Size of the output vector
            (i.e the number of total characters in your language, including EOS and SOS).
            Default: ``None``.
        dropout (float, optional): The probability of the dropout layers.
            Default: 0.1.
        max_length (int, optional): The maximum number of characters that can occur in an image.
            Default: 50.
    """

    def __init__(self,
                 encoder_input=1536,
                 encoder_hidden=128,
                 encoder_nlayers=2,
                 decoder_hidden=128,
                 decoder_nlayers=2,
                 decoder_output=None,
                 dropout=0.1,
                 max_length=MAX_LENGTH,
                 ):

        # if providing decoder_output as None, they need to use load_data
        # Only provide decoder_output, if using custom data loaders.

        self.max_length = max_length

        self.img_encoder = ImgEncoder()

        self.encoder = Encoder(encoder_input,
                               encoder_hidden,
                               encoder_nlayers)
        self.decoder = AttnDecoderRNN(decoder_hidden,
                                      decoder_output,
                                      decoder_nlayers,
                                      dropout,
                                      max_length)

    def load_data(self,
                  train_dir,
                  train_label_path,
                  val_dir=None,
                  val_label_path=None,
                  transform=None):
        """
        Args:
            train_dir (str): directory containing training images.
            train_label_path (str): path to a file with training labels.
            val_dir (str, optional): directory containing validation images.
                Default: None
            val_label_path (str, optional): path to a file with training labels.
                Default: None
            transform (:class:`~torchvision.transforms`, optional): transforms to be applied on images.
                Default: None       
        """

        self.train_dl = OCRDataLoader(train_dir, train_label_path, transform)

        # if loading data, then make decoder_output
        decoder_output = len(self.train_dl.letter2index)
        self.decoder.embedding = nn.Embedding(
            decoder_output, self.decoder.hidden_size)
        self.decoder.out = nn.Linear(self.decoder.hidden_size, decoder_output)
        self.decoder.output_size = decoder_output

        if val_dir is None or val_label_path is None:
            self.val_dl = None
        else:
            self.val_dl = OCRDataLoader(val_dir, val_label_path, transform)

    def forward_step(self,
                     x,
                     y,
                     criterion,
                     device=None,
                     teacher_forcing_ratio=0.1):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        loss = 0

        out = self.img_encoder(x)
        # conv_output shape will be [1,1536, x']

        enc_hidden = self.encoder.initHidden()

        # for attention
        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=device)

        # pass each of the [1, 1536] vectors to encoder
        for idx in range(out.shape[2]):
            enc_out, enc_hidden = self.encoder(out[:, :, idx], enc_hidden)
            encoder_outputs[idx] = enc_out[0, 0]

        decoder_input = torch.tensor(
            [[[self.train_dl.SOS]]], device=device)

        dec_hidden = enc_hidden

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        target_len = len(y)

        if use_teacher_forcing:
            for di in range(target_len):
                dec_out, dec_hidden, dec_attn = self.decoder(
                    decoder_input, dec_hidden, encoder_outputs)
                loss += criterion(dec_out, y[di])
                decoder_input = y[di]  # teacher forcing
        else:
            for di in range(target_len):
                dec_out, dec_hidden, dec_attn = self.decoder(
                    decoder_input, dec_hidden, encoder_outputs)

                topv, topi = dec_out.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += criterion(dec_out, y[di])

                if decoder_input.item() == self.train_dl.EOS:
                    break

        return loss, target_len

# TODO: before creating optimizers make sure they do not already exist
    def initOptimizers(self,
                       imgenc_optimizer=None,
                       enc_optimizer=None,
                       dec_optimizer=None,
                       learning_rate=0.001,):

        if imgenc_optimizer is None:
            imgenc_optimizer = optim.Adam(
                self.img_encoder.parameters(), lr=learning_rate)
        if enc_optimizer is None:
            enc_optimizer = optim.Adam(
                self.encoder.parameters(), lr=learning_rate)
        if dec_optimizer is None:
            dec_optimizer = optim.Adam(
                self.decoder.parameters(), lr=learning_rate)
        return imgenc_optimizer, enc_optimizer, dec_optimizer

    def train(self,
              n_epochs,
              train_dl=None,
              val_dl=None,
              batch_size=1,
              shuffle=True,
              criterion=None,
              imgenc_optimizer=None,
              enc_optimizer=None,
              dec_optimizer=None,
              learning_rate=0.001,
              save_checkpoints=True,
              checkpoint_freq=None,
              checkpoint_path="./",  # Change to None
              max_to_keep=5,
              device=None,
              ):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.img_encoder.to(device)
        self.encoder.to(device)
        self.decoder.to(device)

        imgenc_optimizer, enc_optimizer, dec_optimizer = self.initOptimizers()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        checkpoint_path = join(checkpoint_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        if checkpoint_freq is None:
            checkpoint_freq = n_epochs//10+1

        for epoch in range(n_epochs):
            if train_dl is None:
                if getattr(self, "train_dl", None) is None:
                    raise Exception(
                        "No DataLoader found. Either pass one in train or use load_data method.")
                else:
                    train_dl = self.train_dl(batch_size, shuffle)
            if val_dl is None:
                if getattr(self, "val_dl", None) is None:
                    val_dl = None
                else:
                    val_dl = self.val_dl(batch_size, shuffle)

            self.img_encoder.train()
            self.encoder.train()
            self.decoder.train()
            total_loss = 0

            # Progress bar
            pbar = tqdm(total=len(train_dl))
            pbar.set_description(f"Epoch: {epoch}. Traininig")

            for i, (x, y) in enumerate(train_dl):

                if x.shape[3] < 16:
                    pbar.update(1)
                    continue

                imgenc_optimizer.zero_grad()
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                # maybe add teacher_force_ratio control
                x = x.to(device)
                y = y.to(device)
                y = y[0]  # only for batch_size=1, look into it

                loss, target_len = self.forward_step(
                    x, y, criterion, device)

                loss.backward()

                imgenc_optimizer.step()
                enc_optimizer.step()
                dec_optimizer.step()

                total_loss = total_loss + (loss.item()/target_len)

                pbar.update()
                pbar.set_postfix(loss=total_loss/(i+1))
            pbar.close()

            if val_dl is not None:
                self.img_encoder.eval()
                self.encoder.eval()
                self.decoder.eval()
                eval_loss = 0

                # validation progress bar
                pbar = tqdm(total=len(val_dl))
                pbar.set_description(f"Epoch: {epoch}. Validating")

                for i, (x, y) in enumerate(val_dl):
                    if x.shape[3] < 16:
                        pbar.update(1)
                        continue

                    x = x.to(device)
                    y = y.to(device)
                    y = y[0]  # only

                    loss, target_len = self.forward_step(
                        x, y, criterion, device)

                    eval_loss = eval_loss + (loss.item()/target_len)

                    pbar.update(1)
                    pbar.set_postfix(loss=eval_loss/(i+1))
                pbar.close()
            if epoch % checkpoint_freq == 0:
                self.save_model(checkpoint_path, epoch)

    def load_model(self, path):
        models = torch.load(path)
        print(self.img_encoder.load_state_dict(models["img_encoder"]))
        print(self.encoder.load_state_dict(models["encoder"]))
        print(self.decoder.load_state_dict(models["decoder"]))
        print("Models Loaded!")

    def save_model(self, path, postfix=0):
        name = "{} {}-{}-{} {}.{}.{}.pt".format(postfix, *localtime()[:6])
        models = {"img_encoder": self.img_encoder.state_dict(),
                  "encoder": self.encoder.state_dict(),
                  "decoder": self.decoder.state_dict()}
        torch.save(models, join(path, name))
        print("Models Saved!")


# TODO: Test each and every module of this file.
