from typing import List, Dict
import matplotlib.pyplot as plt

from model.model_abc import Model

import torch
import torch.nn.functional as F


class LinearBigram(Model):

    def __init__(self, num_neurons: int):
        self.x = []
        self.y = []
        self.x_encoded = None
        self.y_encoded = None
        self.W = None
        self.num_neurons = num_neurons

    def _create_xy(self):
        for w in self.words:
            characters = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(characters, characters[1:]):
                index_ch1 = self.encoder[ch1]
                index_ch2 = self.encoder[ch2]
                self.x.append(index_ch1)
                self.y.append(index_ch2)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

    def _encode_xy(self):
        self.x_encoded = F.one_hot(self.x, len(self.chars))
        self.y_encoded = F.one_hot(self.y, len(self.chars))

    def _initialize_w(self):
        self.W = torch.randn(len(self.chars), self.num_neurons)

    def setup(self, path: str):
        self._read_words(path)
        self._create_encoder()
        self._create_decoder()
        self._create_xy()
        self._initialize_w()


def main():
    model = LinearBigram()
    model.setup(Model.PATH)


if __name__ == "__main__":
    main()
