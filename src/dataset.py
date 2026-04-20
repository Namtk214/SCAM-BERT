"""
PyTorch Dataset classes cho T1 và T4.

T1: Sequence Classification (single-label, 3 classes)
T4: Sequence Classification (multi-label, K tactic classes)

Cả hai đều dùng text_segmented đã qua word segmentation
làm input cho PhoBERT tokenizer.
"""

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ScamT1Dataset(Dataset):
    """
    Dataset cho T1 – Scam Detection (multi-class, single-label).

    Mỗi sample gồm:
    - input_ids, attention_mask: từ tokenizer PhoBERT
    - label: integer label (0=SCAM, 1=AMBIGUOUS, 2=LEGIT)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Dùng text_segmented (đã qua word segmentation)
        encoding = self.tokenizer(
            sample["text_segmented"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label_id"], dtype=torch.long),
        }


class ScamT4Dataset(Dataset):
    """
    Dataset cho T4 – Tactic Classification (multi-label).

    Mỗi sample gồm:
    - input_ids, attention_mask: từ tokenizer PhoBERT
    - labels: multi-hot vector float (K chiều)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["text_segmented"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                sample["label_multi_hot"], dtype=torch.float
            ),
        }
