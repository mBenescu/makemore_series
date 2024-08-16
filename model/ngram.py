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
        self.ngram_counts = None
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
        self.chars.append(".")
        self.chars = sorted(self.chars)
        self.encoder = {s: i for i, s in enumerate(self.chars)}

    def _create_decoder(self) -> None:
        """
        Decodes the integers, by reversing the tuples (key, value) in the encoder
        """
        self.decoder = {i: s for s, i in self.encoder.items()}

    def _create_ngrams_counts(self) -> None:
        """
        Stores the number of times a sequence of n characters has been observed in an n-dimensional tensor

        Notes: The n is the integer of the ngram, for a bigram model, n == 2
        """
        self.ngram_counts = torch.zeros(size=[len(self.chars)] * self.n, dtype=torch.int32)
        for w in self.words:
            characters = ["."] + list(w) + ["."]
            iterators = [characters[i:] for i in range(len(characters))]
            for seq in iterators:
                ngram = seq[:self.n]
                if len(ngram) == self.n:
                    ch_indices = [self.encoder[ch] for ch in ngram]
                    self.ngram_counts[tuple(ch_indices)] += 1

    def _create_probs(self) -> None:
        """
        Creates an n-dimensional tensor (self.probs) that stores the probability distributions of observing any
        character after an n-1 sequence of characters

        E.g. In a bigram model (assuming a, b and c are encoded as 1, 2, 3, respectively), self.probs[1][2] represents
        the probability of observing the character "b" after "a". In a trigram, self.probs[1][2][3] represents
        the probability of observing the character "c" after the sequence "ab", etc

        Note: The ngram_count are artificially increased by the smooth_factor to avoid zero probs/ infinite nll
        """

        self.probs = self.ngram_counts.float() + self.smooth_factor
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
        plt.imshow(self.ngram_counts, cmap='Blues')

        font_size = 5

        for i in range(len(self.chars)):
            for j in range(len(self.chars)):
                ch_str = self.decoder[i] + self.decoder[j]
                plt.text(j, i, ch_str, ha="center", va="bottom", color="gray", fontsize=font_size + 4)
                plt.text(j, i, str(self.ngram_counts[i, j].item()), ha="center", va="top", color="gray",
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
            iterators = [characters[i:] for i in range(len(characters))]
            for seq in iterators:
                ngram = seq[:self.n]
                if len(ngram) == self.n:
                    ch_indices = [self.encoder[ch] for ch in ngram]
                    prob = self.probs[tuple(ch_indices)]
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
        self._create_ngrams_counts()
        self._create_probs()

    def __call__(self, g: torch.Generator = None) -> str:
        """
        Samples from model given its probability distributions

        :param g: Generator used in sampling from the multinomial
        :return: The string generated by the model
        """
        ch_indices = [0] * (self.n - 1)
        generated_str = ""
        while True:
            # Extract the character probabilities from the probs matrix
            probs = self.probs[tuple(ch_indices)]
            ch_index = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            # Decode the sampled index
            ch = self.decoder[ch_index]

            # Slide the ch_indices to move to the next character in the probability tensor
            ch_indices = ch_indices[1:] + [ch_index]

            generated_str += ch
            # Break when the sampled character is the ending character (".")
            if ch == ".":
                return generated_str[:-1]


# TODO: Generalise the bigram to ngram model

def main():
    bigram = NgramModel(n=2)
    bigram.setup(PATH)

    trigram = NgramModel(n=3)
    trigram.setup(PATH)

    fourgram = NgramModel(n=4)
    fourgram.setup(PATH)

    g = torch.Generator().manual_seed(NgramModel.SEED)

    print(f"{bigram.get_avg_nll(bigram.words)=}, {trigram.get_avg_nll(trigram.words)=},"
          f" {fourgram.get_avg_nll(fourgram.words)=}")

    for _ in range(10):
        print(f"{bigram(g=g)=}, {trigram(g=g)=}, {fourgram(g=g)=}")


if __name__ == "__main__":
    main()
