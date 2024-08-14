from typing import List, Dict
import torch
import sys
import matplotlib.pyplot as plt

PATH = '..\\data\\names\\names.txt'


class BigramModel:
    NUM_CHARACTERS = 27
    SEED = 2147483647

    def __init__(self, smooth_factor: int = 1):
        self.words = None
        self.encoder = None
        self.decoder = None
        self.bigram_counts = torch.zeros(BigramModel.NUM_CHARACTERS, BigramModel.NUM_CHARACTERS
                                         , dtype=torch.int32)
        self.probs = None
        self.avg_nll = None
        self.smooth_factor = smooth_factor

    def _read_words(self, path: str) -> None:
        self.words = open(path, 'r').read().splitlines()

    def _create_encoder(self) -> None:
        chars = list(set("".join(self.words)))
        chars = sorted(chars)
        self.encoder = {s: i + 1 for i, s in enumerate(chars)}
        self.encoder["."] = 0

    def _create_decoder(self) -> None:
        self.decoder = {i: s for s, i in self.encoder.items()}

    def _create_bigrams_counts(self) -> None:
        for w in self.words:
            characters = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(characters, characters[1:]):
                i_ch1 = self.encoder[ch1]
                i_ch2 = self.encoder[ch2]
                self.bigram_counts[i_ch1, i_ch2] += 1

    def _create_probs(self) -> None:
        # Artificially add counts to the matrix (smoothing) to avoid zero probs/ infinite nll
        self.probs = self.bigram_counts.float() + self.smooth_factor

        # Normalize the counts by their sum to get probability distributions
        self.probs /= self.probs.sum(dim=1, keepdim=True)
        # probs.shape =                               (27, 27)  |
        #                                                       |=> The broadcasting will copy the column
        # self.probs.sum(dim=1, keepdim=True).shape = (27,  1)  |   27 times and perform element-wise division

    def display_bigrams_counts(self) -> None:
        plt.figure(figsize=(16, 16))
        plt.imshow(self.bigram_counts, cmap='Blues')

        font_size = 5

        for i in range(BigramModel.NUM_CHARACTERS):
            for j in range(BigramModel.NUM_CHARACTERS):
                ch_str = self.decoder[i] + self.decoder[j]
                plt.text(j, i, ch_str, ha="center", va="bottom", color="gray", fontsize=font_size + 4)
                plt.text(j, i, str(self.bigram_counts[i, j].item()), ha="center", va="top", color="gray",
                         fontsize=font_size)
        plt.axis("off")
        plt.show()

    def get_avg_nll(self, words):
        """
        GOAL: Maximize likelihood of the given examples
         == Maximize the log likelihood (log is monotonically increasing)
         == Minimize the negative log likelihood
         == Minimize the average negative log likelihood

        :return: The average negative log likelihood across all examples within the dataset
        """
        log_likelihood = 0
        n = 0
        for w in words:
            characters = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(characters, characters[1:]):
                index_ch1 = self.encoder[ch1]
                index_ch2 = self.encoder[ch2]
                prob = self.probs[index_ch1][index_ch2]
                log_likelihood += torch.log(prob)
                n += 1
        avg_nll = -log_likelihood / n
        self.avg_nll = avg_nll
        return avg_nll

    def setup(self, path) -> None:
        self._read_words(path)
        self._create_encoder()
        self._create_decoder()
        self._create_bigrams_counts()
        self._create_probs()

    def __call__(self, g: torch.Generator = None) -> str:
        ch_index = 0
        generated_str = ""
        while True:
            # Extract the character probabilities from the probs matrix
            probs = self.probs[ch_index]
            # Sample given the probabilities
            ch_index = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            # Decode the sampled index
            ch = self.decoder[ch_index]
            generated_str += ch
            # Break when the sampled character is the ending character (".")
            if ch == ".":
                return generated_str


# TODO: Add typehints for the future methods

def main():
    model = BigramModel()
    model.setup(PATH)
    g = torch.Generator().manual_seed(BigramModel.SEED)

    print(model.get_avg_nll(model.words))

    # model.display_bigrams_counts()
    # for _ in range(5):
    #     print(model(g=g))


if __name__ == "__main__":
    main()
