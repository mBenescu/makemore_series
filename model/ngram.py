from typing import List, Dict
import torch
import sys
import matplotlib.pyplot as plt

PATH = '..\\data\\names\\names.txt'


class NgramModel:
    SEED = 2147483647

    def __init__(self, n: int, smooth_factor: int = 1):
        self.n = n
        self.smooth_factor = smooth_factor
        self.words = None
        self.chars = None
        self.encoder = None
        self.decoder = None
        self.bigram_counts = None
        self.probs = None
        self.avg_nll = None

    def _read_words(self, path: str) -> None:
        """
        Reads the words at path
        :param path:  location of the file
        """
        self.words = open(path, 'r').read().splitlines()

    def _create_encoder(self) -> None:
        """
        Encodes the characters in a dictionary (self.encoder) where the key is the character and value is a numerical
        integer between 0 and 27

        Notes: The character "." denotes both the start and the ending of a word and it is encoded as 0, the rest of the
        characters take increasing values from 1, up to 27, in an alphabetical order (i.e. a -> 1, b -> 2, etc)
        """
        self.chars = list(set("".join(self.words)))
        self.chars = sorted(self.chars)
        self.encoder = {s: i + 1 for i, s in enumerate(self.chars)}
        self.encoder["."] = 0

    def _create_decoder(self) -> None:
        """
        Decodes the integers, by reversing the tuples (key, value) in the encoder
        """
        self.decoder = {i: s for s, i in self.encoder.items()}

    def _create_bigrams_counts(self) -> None:
        """
        Stores the number of times a sequence of n characters has been observed in an n-dimensional tensor

        Notes: The n is the integer of the ngram, for a bigram model, n == 2
        """
        self.bigram_counts = torch.zeros(size=[len(self.chars)] * self.n, dtype=torch.int32)
        for w in self.words:
            characters = ["."] + list(w) + ["."]
            iterators = [characters[i:] for i in range(self.n)]
            for seq in zip(*iterators):
                ch_indices = [self.encoder[ch] for ch in seq]
                self.bigram_counts[tuple(ch_indices)] += 1

    def _create_probs(self) -> None:
        """
        Creates an n-dimensional tensor (self.probs) that stores the probability distributions of observing any
        character after an n-1 sequence of characters

        E.g. In a bigram model (assuming a, b and c are encoded as 1, 2, 3, respectively), self.probs[1][2] represents
        the probability of observing the character "b" after "a". In a trigram, self.probs[1][2][3] represents
        the probability of observing the character "c" after the sequence "ab", etc

        Note: The ngram_count are artificially increased by the smooth_factor to avoid zero probs/ infinite nll
        """

        self.probs = self.bigram_counts.float() + self.smooth_factor
        # Normalize the counts by their sum to get probability distributions
        self.probs /= self.probs.sum(dim=(self.n - 1), keepdim=True)

        # For bigram:
        # probs.shape =                               (27, 27)  |
        #                                                       |=> The broadcasting will copy the column
        # self.probs.sum(dim=1, keepdim=True).shape = (27,  1)  |   27 times and perform element-wise division

    def display_bigrams_counts(self) -> None:
        """
        Creates a figure containing the bigram counts

        Note: Only works for bigram models
        """
        plt.figure(figsize=(16, 16))
        plt.imshow(self.bigram_counts, cmap='Blues')

        font_size = 5

        for i in range(len(self.chars)):
            for j in range(len(self.chars)):
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
        """
        Calls the methods in the right order to use the model
        :param path: location to read the words from
        """
        self._read_words(path)
        self._create_encoder()
        self._create_decoder()
        self._create_bigrams_counts()
        self._create_probs()

    def __call__(self, g: torch.Generator = None) -> str:
        """
        Samples from model given its probability distributions

        :param g: Generator used in sampling from the multinomial
        :return: The string generated by the model
        """
        ch_index = 0
        generated_str = ""
        while True:
            # Extract the character probabilities from the probs matrix
            probs = self.probs[ch_index]
            ch_index = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            # Decode the sampled index
            ch = self.decoder[ch_index]
            generated_str += ch
            # Break when the sampled character is the ending character (".")
            if ch == ".":
                return generated_str


# TODO: Generalise the bigram to ngram model

def main():
    model = NgramModel()
    model.setup(PATH)
    g = torch.Generator().manual_seed(NgramModel.SEED)

    print(model.get_avg_nll(model.words))

    # model.display_bigrams_counts()
    # for _ in range(5):
    #     print(model(g=g))


if __name__ == "__main__":
    main()
