from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from model.model_abc import Model

import torch
import torch.nn.functional as F


class MLP(Model):

    def __init__(self, context_length: int = 3, no_emb_dim: int = 2):
        super().__init__()
        self.context_length = context_length
        self.no_emb_dim = no_emb_dim
        self.emb_table = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _initialize_params(self):
        self.emb_table = torch.randn(len(self.chars), self.no_emb_dim)
        # self.W1 = torch.randn()

    def create_datasets(self, words: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for w in words:
            context = [0] * self.context_length
            characters = w + "."
            for ch in characters:
                ch_index = self.encoder[ch]
                X.append(context)
                Y.append(ch_index)
                # print(''.join(self.decoder[enc_ch] for enc_ch in context), "--->", self.decoder[ch_index])
                # print(''.join(str(context)), "--->", ch_index)
                context = context[1:] + [ch_index]  # crop and append the context
        X, Y = torch.tensor(X), torch.tensor(Y)
        return X, Y

    def setup(self, path: str):
        self._read_words(path)
        self._create_encoder()
        self._create_decoder()

    def __call__(self, g: torch.Generator) -> str:
        pass


def main():
    mlp = MLP()
    mlp.setup(Model.PATH)
    X, Y = mlp.create_datasets(mlp.words[:4])
    print(X.shape)
    print(Y.shape)

if __name__ == "__main__":
    main()