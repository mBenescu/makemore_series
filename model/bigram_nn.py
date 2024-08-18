from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from model.model_abc import Model

import torch
import torch.nn.functional as F

from model.ngram import NgramModel


class LinearBigram(Model):

    def __init__(self, num_neurons: int, lr: float = 0.1, reg_fact: int = 0.01):
        super().__init__()
        self.num_neurons = num_neurons
        self.lr = lr
        self.reg_fact = reg_fact
        self.x = None
        self.y = None
        self.W = None
        self.probs = None
        self.loss = None

    def _create_xy(self, words: str) -> None:
        self.x = []
        self.y = []
        for w in words:
            characters = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(characters, characters[1:]):
                index_ch1 = self.encoder[ch1]
                index_ch2 = self.encoder[ch2]
                self.x.append(index_ch1)
                self.y.append(index_ch2)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x_encoded = F.one_hot(x, len(self.chars)).float()
        return x
        # return x_encoded

    def _initialize_w(self, g: torch.Generator = None):
        self.W = torch.randn(len(self.chars), self.num_neurons, generator=g, requires_grad=True)

    def _get_probs(self, x):
        x_encoded = self._encode(x)
        # logits = x_encoded.float() @ self.W
        logits = self.W[x_encoded]
        counts = logits.exp()
        self.probs = counts / counts.sum(dim=1, keepdim=True)

    def forward(self):
        self._get_probs(self.x)
        self.loss = -self.probs[torch.arange(self.x.nelement()), self.y].log().mean() + self.reg_fact * (
                self.W ** 2).mean()

    def backward(self):
        self.W.grad = None
        self.loss.backward()

    def sgd(self):
        with torch.no_grad():
            self.W += -self.lr * self.W

    def train(self, num_iter):
        for i in range(num_iter):
            self.forward()

            # if i % 10 == 0:
            print(f"loss at {i=} = {self.loss}")

            self.backward()
            self.sgd()

    def __call__(self, g: torch.Generator = None) -> str:
        generated_str = ""
        ch_index = 0
        while True:
            x = torch.tensor([ch_index])
            self._get_probs(x)
            ch_index = torch.multinomial(self.probs, num_samples=1, replacement=True, generator=g).item()
            ch = self.decoder[ch_index]
            if ch == ".":
                return generated_str
            generated_str += ch

    def setup(self, path: str):
        self._read_words(path)
        self._create_encoder()
        self._create_decoder()
        self._initialize_w(g=torch.Generator().manual_seed(Model.SEED))
        self._create_xy(self.words)


def main():
    model = LinearBigram(num_neurons=27, lr=0.1)
    model.setup(Model.PATH)
    model.train(100)

    bigram = NgramModel(n=2)
    bigram.setup(Model.PATH)
    bigram.get_avg_nll(bigram.words)

    print(f"{model.loss=}, {bigram.avg_nll=}")


if __name__ == "__main__":
    main()
