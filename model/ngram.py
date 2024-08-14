from typing import List, Dict
import torch
import sys
import matplotlib.pyplot as plt

CHARACTER_NO = 28


def read_words() -> List[str]:
    words = open('..\\data\\names\\names.txt', 'r').read().splitlines()
    return words


def print_words_info(words: List[str]):
    print(words[:10])
    print(len(words))
    print("min_len = ", min(len(w) for w in words))
    print("max_len = ", max(len(w) for w in words))


def get_stoi(words: List[str]) -> Dict[str, int]:
    chars = list(set("".join(words)))
    stoi = {s: i for i, s in enumerate(chars)}
    stoi["<S>"] = len(stoi)
    stoi["<E>"] = len(stoi)
    return stoi


def get_bigrams_counts(words: List[str], stoi: Dict[str, int]) -> Dict[str, int]:
    N = torch.zeros(CHARACTER_NO, CHARACTER_NO, dtype=torch.int32)
    for w in words:
        characters = ["<S>"] + list(w) + ["<E>"]
        for ch1, ch2 in zip(characters, characters[1:]):
            i_ch1 = stoi[ch1]
            i_ch2 = stoi[ch2]
            N[i_ch1, i_ch2] += 1
    return N


def main():
    words = read_words()
    # print_words_info(words)
    # biagram_counts = get_bigrams_counts(words)

    # print(sorted(biagram_counts.items(), key=lambda kv: -kv[1]))

    stoi = get_stoi(words)
    print(stoi)
    print(get_bigrams_counts(words, stoi))


if __name__ == "__main__":
    main()
