import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch  # 新增
from transformers import AutoTokenizer

from .inlegalner_dataset import InLegalNERDataset, build_label_vocab


def ner_collate_fn(batch):
    """
    自定义的 collate 函数：
    - 对 batch 中每个样本的各个字段分别处理
    - 如果字段是 Tensor，则 clone().detach() 后再 stack，避免共享 storage 引发的 resize 报错
    - 如果字段不是 Tensor（比如原始 tokens 列表等），则原样组成 list 返回
    """
    if len(batch) == 0:
        return {}

    collated = {}
    keys = batch[0].keys()

    for k in keys:
        values = [example[k] for example in batch]

        if torch.is_tensor(values[0]):
            collated[k] = torch.stack(
                [v.clone().detach() for v in values], dim=0
            )
        else:
            collated[k] = values

    return collated


class LegalNERDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Legal NER (InLegalNER-style) data.

    Expected directory structure:

        data/raw/inlegalner/
            train.conll
            dev.conll
            test.conll
    """

    def __init__(
        self,
        data_dir: str = "data/raw/inlegalner",
        pretrained_model_name: str = "nlpaueb/legal-bert-base-uncased",
        max_length: int = 256,
        train_file: str = "train.conll",
        dev_file: str = "dev.conll",
        test_file: str = "test.conll",
        batch_size: int = 16,
        num_workers: int = 0,  # 默认先用 0，稳定后你可以改回 4
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = None
        self.label2id = None
        self.id2label = None

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """
        Called only from a single process.

        - Download tokenizer if needed.
        - (Optional) You can also put dataset download/conversion here.
        """
        AutoTokenizer.from_pretrained(self.pretrained_model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called on every process.

        We:
        - instantiate tokenizer
        - build label vocabulary from train file
        - create train / dev / test datasets
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        train_path = os.path.join(self.data_dir, self.train_file)
        dev_path = os.path.join(self.data_dir, self.dev_file)
        test_path = os.path.join(self.data_dir, self.test_file)

        # Build label vocab from training set
        if self.label2id is None or self.id2label is None:
            self.label2id, self.id2label = build_label_vocab(train_path)

        if stage in (None, "fit"):
            self.train_dataset = InLegalNERDataset(
                file_path=train_path,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                max_length=self.max_length,
            )
            self.dev_dataset = InLegalNERDataset(
                file_path=dev_path,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                max_length=self.max_length,
            )

        if stage in (None, "test", "predict"):
            self.test_dataset = InLegalNERDataset(
                file_path=test_path,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                max_length=self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ner_collate_fn,  # 使用自定义 collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ner_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ner_collate_fn,
        )

