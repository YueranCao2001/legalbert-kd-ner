import os
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def read_conll_file(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read a CoNLL-style file where each non-empty line is:
        token TAG
    and sentences are separated by blank lines.

    Returns:
        sentences: List of token lists
        labels:    List of tag lists
    """
    sentences: List[List[str]] = []
    labels: List[List[str]] = []

    tokens: List[str] = []
    tags: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            tag = parts[-1]
            tokens.append(token)
            tags.append(tag)

    if tokens:
        sentences.append(tokens)
        labels.append(tags)

    return sentences, labels


def build_label_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Scan a CoNLL file and build label2id / id2label mappings.
    We only need to scan the training file.
    """
    _, all_labels = read_conll_file(path)

    label_set = set()
    for sent_labels in all_labels:
        label_set.update(sent_labels)

    label_list = sorted(label_set)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


class InLegalNERDataset(Dataset):
    """
    A torch.utils.data.Dataset for CoNLL-style InLegalNER data.

    It handles:
    - tokenization with a Hugging Face tokenizer
    - alignment of word-level labels to subword tokens
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        self.sentences, self.labels = read_conll_file(file_path)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sentences[idx]
        tags = self.labels[idx]

        # 关键修改：padding="max_length"，保证所有样本长度一致
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_offsets_mapping=False,
        )

        word_ids = encoding.word_ids()
        label_ids: List[int] = []

        previous_word_id: Optional[int] = None
        for word_id in word_ids:
            if word_id is None:
                # special tokens + padding → ignore in loss
                label_ids.append(-100)
            else:
                if word_id != previous_word_id:
                    tag = tags[word_id]
                    label_ids.append(self.label2id[tag])
                else:
                    label_ids.append(-100)
            previous_word_id = word_id

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
        return item
