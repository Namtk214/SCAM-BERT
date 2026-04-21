"""
Cấu hình trung tâm cho pipeline fine-tuning PhoBERT.
Tách riêng config T1 (Scam Detection) và T4 (Tactic Classification).
"""

from dataclasses import dataclass, field
from typing import List
import os

# ============================================================
# Đường dẫn gốc
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Taxonomy nhãn – phải cố định trước khi train (mục 4)
# ============================================================
T1_LABELS: List[str] = ["SCAM", "AMBIGUOUS", "LEGIT"]

TACTIC_LABELS: List[str] = [
    "AUTHORITY",
    "SCARCITY",
    "SOCIAL_PROOF",
    "LIKING",
    "RECIPROCITY",
    "COMMITMENT",
    "THREAT_LEGAL",
    "THREAT_FINANCIAL",
    "THREAT_SOCIAL",
    "ISOLATION",
    "OVERWHELM",
    "HELP",
    "TRUST_BUILD_REVEAL",
    "CONFESSION_ELICIT",
    "REFRAME",
    "INFO_REQUEST",
    "ACTION_REQUEST",
    "DEFLECT_EXTERNAL",
    "DEFLECT_MINIMIZE",
]

T1_LABEL2ID = {label: idx for idx, label in enumerate(T1_LABELS)}
T1_ID2LABEL = {idx: label for idx, label in enumerate(T1_LABELS)}

T4_LABEL2ID = {label: idx for idx, label in enumerate(TACTIC_LABELS)}
T4_ID2LABEL = {idx: label for idx, label in enumerate(TACTIC_LABELS)}


# ============================================================
# Cấu hình huấn luyện – theo khuyến nghị paper PhoBERT (mục 9)
# ============================================================
@dataclass
class TrainingConfig:
    """Hyperparameters khởi điểm dựa trên paper PhoBERT."""

    # Model
    model_name: str = "vinai/phobert-base"
    max_seq_length: int = 256

    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.1

    # Training
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Evaluation & Checkpointing
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    save_total_limit: int = 3

    # Data
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Paths
    raw_data_path: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "raw_conversations.json")
    )
    processed_data_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "processed")
    )
    output_dir_t1: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "outputs", "t1_scam_detection")
    )
    output_dir_t4: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "outputs", "t4_tactic_classification")
    )
    vncorenlp_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "vncorenlp")
    )

    # T4-specific
    t4_threshold: float = 0.5
