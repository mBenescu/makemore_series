from abc import ABC, abstractmethod


class Model(ABC):
    PATH = '..\\data\\names\\names.txt'

    def __init__(self):
        self.words = None
        self.chars = None
        self.encoder = None
        self.decoder = None

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

    @abstractmethod
    def setup(self, path: str):
        pass
