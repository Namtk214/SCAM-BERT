"""
Training script cho T4 – Tactic Classification.

Quy trình (mục 11):
  1. Load preprocessed T4 data (text_segmented, single turn).
  2. Tokenize bằng PhoBERT tokenizer.
  3. Fine-tune PhoBERT + multi-label head (K tactic).
  4. Loss: BCEWithLogitsLoss (mục 8.2).
  5. Sigmoid → threshold trên validation.
  6. Đánh giá: Micro-F1, Macro-F1, Per-label F1, Exact match.

Model: AutoModelForSequenceClassification (mục 8.2)
      Đầu ra K logits, mỗi chiều dùng sigmoid độc lập.
Mode:  Fully fine-tuning (mục 8.3)
"""

import os
import sys
import numpy as np

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from config import (
    TACTIC_LABELS,
    T4_ID2LABEL,
    T4_LABEL2ID,
    TrainingConfig,
)
from dataset import ScamT4Dataset
from metrics import compute_t4_metrics, print_t4_report


# ============================================================
# Custom Trainer cho multi-label loss (mục 8.2)
# ============================================================
class MultiLabelTrainer(Trainer):
    """
    Trainer tùy biến dùng BCEWithLogitsLoss cho multi-label.
    Transformers mặc định dùng CrossEntropy,
    nhưng T4 là multi-label nên cần override compute_loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


# ============================================================
# Threshold tuning trên validation (mục 11)
# ============================================================
def tune_threshold(trainer: Trainer, val_dataset, thresholds=None):
    """
    Tìm threshold tốt nhất trên validation set.
    Mặc định 0.5 chỉ là điểm xuất phát (mục 11).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    output = trainer.predict(val_dataset)
    logits = output.predictions
    labels = output.label_ids
    probs = 1 / (1 + np.exp(-logits))

    best_threshold = 0.5
    best_f1 = 0.0

    for th in thresholds:
        preds = (probs >= th).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    print(f"\n  Best threshold: {best_threshold:.2f} (Macro-F1 = {best_f1:.4f})")
    return best_threshold


# ============================================================
# Main training function
# ============================================================
def train_t4(cfg: TrainingConfig):
    """Huấn luyện mô hình T4."""

    print("=" * 60)
    print("T4 – TACTIC CLASSIFICATION TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------------
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------
    train_dataset = ScamT4Dataset(
        os.path.join(cfg.processed_data_dir, "t4_train.json"),
        tokenizer,
        cfg.max_seq_length,
    )
    val_dataset = ScamT4Dataset(
        os.path.join(cfg.processed_data_dir, "t4_val.json"),
        tokenizer,
        cfg.max_seq_length,
    )
    test_dataset = ScamT4Dataset(
        os.path.join(cfg.processed_data_dir, "t4_test.json"),
        tokenizer,
        cfg.max_seq_length,
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # --------------------------------------------------------
    # 3. Load model – multi-label head (mục 8.2)
    # --------------------------------------------------------
    num_labels = len(TACTIC_LABELS)
    print(f"Loading model: {cfg.model_name} (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=T4_ID2LABEL,
        label2id=T4_LABEL2ID,
        problem_type="multi_label_classification",
    )
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --------------------------------------------------------
    # 4. Training arguments
    # --------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir_t4,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        adam_epsilon=cfg.adam_epsilon,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        logging_dir=os.path.join(cfg.output_dir_t4, "logs"),
        logging_steps=10,
        report_to="none",
        fp16=False,
    )

    # --------------------------------------------------------
    # 5. Trainer (custom loss)
    # --------------------------------------------------------
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda ep: compute_t4_metrics(ep, cfg.t4_threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------
    print("\nBắt đầu huấn luyện T4...")
    train_result = trainer.train()
    print(f"\nTraining hoàn tất! Loss cuối: {train_result.training_loss:.4f}")

    # Lưu model
    trainer.save_model(os.path.join(cfg.output_dir_t4, "best_model"))
    tokenizer.save_pretrained(os.path.join(cfg.output_dir_t4, "best_model"))

    # --------------------------------------------------------
    # 7. Tune threshold trên validation (mục 11)
    # --------------------------------------------------------
    print("\nTuning threshold trên validation set...")
    best_threshold = tune_threshold(trainer, val_dataset)

    # --------------------------------------------------------
    # 8. Evaluation trên test set
    # --------------------------------------------------------
    print("\nĐánh giá trên test set:")
    test_output = trainer.predict(test_dataset)

    # Dùng threshold đã tune
    test_logits = test_output.predictions
    test_labels = test_output.label_ids
    print_t4_report(test_labels, test_logits, threshold=best_threshold)

    # In metrics
    test_metrics = compute_t4_metrics(
        (test_logits, test_labels), threshold=best_threshold
    )
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\nModel T4 đã lưu tại: {os.path.join(cfg.output_dir_t4, 'best_model')}")
    print(f"Best threshold: {best_threshold:.2f}")
    return trainer, best_threshold


# ============================================================
if __name__ == "__main__":
    cfg = TrainingConfig()
    if "--small" in sys.argv:
        cfg.per_device_train_batch_size = 8
        cfg.per_device_eval_batch_size = 16
        cfg.num_train_epochs = 10
    train_t4(cfg)
