import random
import re
from typing import List

import nltk
from nltk.corpus import wordnet


# 如果第一次运行，需要下载 wordnet
try:
    _ = wordnet.synsets("test")
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


LEGAL_PUNCT = [";", ":", "/", "-", "—", "(", ")", "[", "]"]


def synonym_replacement(word: str) -> str:
    syns = wordnet.synsets(word)
    if not syns:
        return word
    lemmas = syns[0].lemmas()
    if not lemmas:
        return word
    new = lemmas[0].name().replace("_", " ")
    if new.lower() == word.lower():
        return word
    return new


def random_swap(words: List[str]) -> List[str]:
    if len(words) < 2:
        return words
    i, j = random.sample(range(len(words)), 2)
    words = words.copy()
    words[i], words[j] = words[j], words[i]
    return words


def random_deletion(words: List[str], p=0.1):
    if len(words) == 1:
        return words
    new_words = []
    for w in words:
        if random.random() > p:
            new_words.append(w)
    if not new_words:
        return [words[random.randint(0, len(words) - 1)]]
    return new_words


def char_noise(word: str) -> str:
    if len(word) <= 3:
        return word
    i = random.randint(0, len(word) - 2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]


def legal_punc_noise(word: str) -> str:
    if random.random() < 0.1:
        return random.choice(LEGAL_PUNCT)
    return word


def augment_sentence(tokens: List[str], labels: List[str]) -> (List[str], List[str]):
    assert len(tokens) == len(labels)

    choice = random.choice(["syn", "swap", "del", "char", "punc"])

    if choice == "syn":
        new_tokens = [
            synonym_replacement(w) if l.startswith("O") else w
            for w, l in zip(tokens, labels)
        ]
        return new_tokens, labels

    elif choice == "swap":
        new = random_swap(tokens)
        return new, labels

    elif choice == "del":
        new = random_deletion(tokens)
        # 删除会破坏标签对齐 → 返回原句
        return tokens, labels

    elif choice == "char":
        new_tokens = [
            char_noise(w) if l.startswith("O") else w
            for w, l in zip(tokens, labels)
        ]
        return new_tokens, labels

    elif choice == "punc":
        new_tokens = [
            legal_punc_noise(w) if l.startswith("O") else w
            for w, l in zip(tokens, labels)
        ]
        return new_tokens, labels

    return tokens, labels


def augment_dataset(sentences, labels, num_aug=10):
    aug_sents = []
    aug_labels = []

    for s, l in zip(sentences, labels):
        for _ in range(num_aug):
            ns, nl = augment_sentence(s, l)
            aug_sents.append(ns)
            aug_labels.append(nl)

    return aug_sents, aug_labels
