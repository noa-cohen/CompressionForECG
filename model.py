import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import ECGdataset, create_train_validation_loaders
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# from training import AutoEncoderTrainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_debug = False


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


class Encoder(nn.Module):

    def __init__(
            self,
            in_size: int = 1,
            out_classes: int = 1,
            channels: list = [],
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        super(Encoder, self).__init__()

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        # Encoder
        layers = []
        # input is ECG window of size 2000
        layers.append(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(num_features=32))
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(num_features=16))
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(nn.Conv1d(in_channels=16, out_channels=64, kernel_size=11, padding=5))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=13, padding=6))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=32, out_channels=1, kernel_size=7, padding=3))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2))

        if _debug:
            for i in range(len(layers)):
                # print(f'i={i}, 2*i+1={2*i+1} len(list)={len(layers)}')
                layers.insert(2 * i + 1, PrintLayer())

        self.layer_sequence = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_sequence(x)
        return x


class Decoder(nn.Module):
    '''
    output
    output is restored ECG window of size 2000
    '''

    def __init__(
            self,
            in_size: int = 1,
            out_classes: int = 1,
            channels: list = [],
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        super(Decoder, self).__init__()

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        layers = []
        linear_layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=13, padding=6))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Upsample(scale_factor=2))
        layers.append(nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        linear_layers.append(nn.Linear(in_features=992, out_features=2000))
        linear_layers.append(nn.Sigmoid())

        if _debug:
            for i in range(len(layers)):
                layers.insert(2 * i + 1, PrintLayer())

        self.layer_sequence = nn.Sequential(*layers)
        self.linear_sequence = nn.Sequential(*linear_layers)

    def forward(self, x, false=False):
        # import pdb
        # pdb.set_trace()
        x = self.layer_sequence(x)
        x = x.view(x.shape[0], -1)
        suc = True
        counter = 0
        while suc:
            try:
                counter += 1
                if counter >= 10:
                    print("Could not pass linear_sequence in decoder more then 10 times")
                    break

                x = self.linear_sequence(x)
                suc = False
            except:
                continue

        return x


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = x.double()
        enc_outputs = self.encoder(x)
        dec_outputs = self.decoder(enc_outputs)
        return dec_outputs


if __name__ == '__main__':
    ecg_window = torch.rand((1, 1, 2000))  # [batch_size, in_channels, len]
    # print(f'Input size: {ecg_window.size()}')

    # Model
    Enc = Encoder()
    Dec = Decoder()
    inst = EncoderDecoder(Enc, Dec)
    # inst = inst.double()
    out = inst.forward(ecg_window)

    diff = ecg_window - out
    diff = diff.squeeze().tolist()

    print('Done.')
