"""
Preprocessing pipeline cho PhoBERT fine-tuning.

Quy trình (theo tài liệu):
  1. Đọc raw JSON conversations.
  2. Chuẩn hóa speaker → Speaker_A / Speaker_B (mục 5.1).
  3. Chuẩn hóa văn bản: khoảng trắng, Unicode, dấu câu (mục 5.2).
  4. Word segmentation bằng RDRSegmenter/VnCoreNLP (mục 6).
  5. Chuyển đổi sang schema huấn luyện T1 và T4 (mục 3.2, 3.3).
  6. Chia tập theo conversation_id (mục 7).
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Tuple

import py_vncorenlp

from config import (
    TACTIC_LABELS,
    T1_LABEL2ID,
    T4_LABEL2ID,
    TrainingConfig,
)


# ============================================================
# 1. Text Cleaning (mục 5.2)
# ============================================================
def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode NFC cho tiếng Việt."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """
    Chuẩn hóa văn bản:
    - Loại ký tự điều khiển.
    - Chuẩn hóa khoảng trắng / xuống dòng.
    - Giữ dấu câu tiếng Việt thông thường.
    """
    text = normalize_unicode(text)
    # Bỏ ký tự điều khiển (trừ newline)
    text = re.sub(r"[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Chuẩn hóa khoảng trắng liên tiếp
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2. Speaker Neutralization (mục 5.1)
# ============================================================
def neutralize_speakers(messages: List[Dict]) -> List[Dict]:
    """
    Đổi speaker_role thành Speaker_A, Speaker_B, ... theo thứ tự xuất hiện.
    Tránh rò rỉ nhãn qua tên role (scammer/victim).
    """
    role_map = {}
    counter = 0
    neutralized = []
    for msg in messages:
        role = msg["speaker_role"]
        if role not in role_map:
            role_map[role] = f"Speaker_{chr(65 + counter)}"  # A, B, C, ...
            counter += 1
        new_msg = dict(msg)
        new_msg["speaker_neutral"] = role_map[role]
        neutralized.append(new_msg)
    return neutralized


# ============================================================
# 3. Word Segmentation (mục 6)
# ============================================================
class WordSegmenter:
    """Wrapper cho VnCoreNLP RDRSegmenter."""

    def __init__(self, vncorenlp_dir: str):
        """
        Parameters
        ----------
        vncorenlp_dir : str
            Thư mục chứa VnCoreNLP models.
            Dùng py_vncorenlp.download_model(save_dir=...) để tải.
        """
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=vncorenlp_dir,
        )

    def segment(self, text: str) -> str:
        """Trả về chuỗi đã word-segment."""
        result = self.rdrsegmenter.word_segment(text)
        # py_vncorenlp trả về list các câu đã segment
        if isinstance(result, list):
            return " ".join(result)
        return result


# ============================================================
# 4. Conversion: Raw → T1 Schema (mục 3.2)
# ============================================================
def build_t1_sample(
    conversation: Dict, segmenter: WordSegmenter
) -> Dict:
    """
    Chuyển đổi 1 conversation thành sample T1.

    Schema T1:
    {
        "conversation_id": "...",
        "label": "SCAM",
        "text_clean": "Speaker_A: ... Speaker_B: ...",
        "text_segmented": "Speaker_A : ... Speaker_B : ..."
    }
    """
    messages = neutralize_speakers(conversation["messages"])

    # Ghép toàn bộ hội thoại theo thứ tự turn
    parts = []
    for msg in messages:
        text = clean_text(msg["text"])
        parts.append(f'{msg["speaker_neutral"]}: {text}')
    text_clean = " ".join(parts)

    # Word segmentation
    text_segmented = segmenter.segment(text_clean)

    return {
        "conversation_id": conversation["conversation_id"],
        "label": conversation["t1_label"],
        "label_id": T1_LABEL2ID[conversation["t1_label"]],
        "text_clean": text_clean,
        "text_segmented": text_segmented,
    }


# ============================================================
# 5. Conversion: Raw → T4 Schema (mục 3.3)
# ============================================================
def build_t4_samples(
    conversation: Dict, segmenter: WordSegmenter
) -> List[Dict]:
    """
    Trích các turn của scammer có t4_labels thành các sample T4.

    Schema T4:
    {
        "conversation_id": "...",
        "turn_id": "...",
        "label_set": ["SA_AUTH", "SA_THREAT"],
        "text_clean": "...",
        "text_segmented": "..."
    }
    """
    samples = []
    for msg in conversation["messages"]:
        if "t4_labels" not in msg or not msg["t4_labels"]:
            continue

        text_clean = clean_text(msg["text"])
        text_segmented = segmenter.segment(text_clean)

        # Multi-hot vector
        multi_hot = [0] * len(TACTIC_LABELS)
        for label in msg["t4_labels"]:
            multi_hot[T4_LABEL2ID[label]] = 1

        samples.append({
            "conversation_id": conversation["conversation_id"],
            "turn_id": msg["turn_id"],
            "label_set": msg["t4_labels"],
            "label_multi_hot": multi_hot,
            "text_clean": text_clean,
            "text_segmented": text_segmented,
        })
    return samples


# ============================================================
# 6. Chia tập theo conversation_id (mục 7)
# ============================================================
def split_by_conversation(
    conversations: List[Dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Chia conversation-level để tránh data leakage.
    Một conversation chỉ nằm trong 1 split duy nhất.
    """
    import random

    random.seed(seed)
    conv_ids = [c["conversation_id"] for c in conversations]
    random.shuffle(conv_ids)

    n = len(conv_ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_ids = set(conv_ids[:n_test])
    val_ids = set(conv_ids[n_test : n_test + n_val])
    train_ids = set(conv_ids[n_test + n_val :])

    id2conv = {c["conversation_id"]: c for c in conversations}

    train = [id2conv[cid] for cid in conv_ids if cid in train_ids]
    val = [id2conv[cid] for cid in conv_ids if cid in val_ids]
    test = [id2conv[cid] for cid in conv_ids if cid in test_ids]

    return train, val, test


# ============================================================
# 7. Pipeline chính
# ============================================================
def run_preprocessing(cfg: TrainingConfig):
    """
    Chạy toàn bộ pipeline preprocessing:
    raw JSON → clean → segment → T1 & T4 samples → chia tập → lưu file.
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Tải VnCoreNLP model nếu chưa có
    if not os.path.exists(os.path.join(cfg.vncorenlp_dir, "VnCoreNLP-1.2.jar")):
        print(f"Đang tải VnCoreNLP models vào {cfg.vncorenlp_dir}...")
        os.makedirs(cfg.vncorenlp_dir, exist_ok=True)
        py_vncorenlp.download_model(save_dir=cfg.vncorenlp_dir)
    segmenter = WordSegmenter(cfg.vncorenlp_dir)

    # Đọc raw data
    print(f"Đọc dữ liệu từ {cfg.raw_data_path}...")
    with open(cfg.raw_data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"  Tổng số conversation: {len(conversations)}")

    # Chia tập theo conversation_id
    train_convs, val_convs, test_convs = split_by_conversation(
        conversations, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    print(f"  Train: {len(train_convs)} | Val: {len(val_convs)} | Test: {len(test_convs)}")

    # Chuyển đổi
    os.makedirs(cfg.processed_data_dir, exist_ok=True)

    for split_name, split_convs in [
        ("train", train_convs),
        ("val", val_convs),
        ("test", test_convs),
    ]:
        t1_samples = []
        t4_samples = []
        for conv in split_convs:
            t1_samples.append(build_t1_sample(conv, segmenter))
            t4_samples.extend(build_t4_samples(conv, segmenter))

        # Lưu T1
        t1_path = os.path.join(cfg.processed_data_dir, f"t1_{split_name}.json")
        with open(t1_path, "w", encoding="utf-8") as f:
            json.dump(t1_samples, f, ensure_ascii=False, indent=2)
        print(f"  T1 {split_name}: {len(t1_samples)} samples → {t1_path}")

        # Lưu T4
        t4_path = os.path.join(cfg.processed_data_dir, f"t4_{split_name}.json")
        with open(t4_path, "w", encoding="utf-8") as f:
            json.dump(t4_samples, f, ensure_ascii=False, indent=2)
        print(f"  T4 {split_name}: {len(t4_samples)} samples → {t4_path}")

    print("\nPreprocessing hoàn tất!")
    return True


# ============================================================
if __name__ == "__main__":
    cfg = TrainingConfig()
    run_preprocessing(cfg)
