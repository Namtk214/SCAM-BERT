"""
Training script cho T1 – Scam Detection.

Quy trình (mục 10):
  1. Load preprocessed T1 data (text_segmented).
  2. Tokenize bằng PhoBERT tokenizer.
  3. Fine-tune PhoBERT + classification head (3 class).
  4. Chọn best checkpoint theo validation macro-F1.
  5. Đánh giá trên test set.

Model: AutoModelForSequenceClassification (mục 8.1)
Loss:  Softmax + CrossEntropy (tự động bởi Transformers)
Mode:  Fully fine-tuning – toàn bộ tham số đều cập nhật (mục 8.3)
"""

import json
import os
import random
import sys
import numpy as np

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
)

from config import (
    T1_LABELS,
    T1_ID2LABEL,
    T1_LABEL2ID,
    TrainingConfig,
)
from dataset import ScamT1Dataset
from metrics import compute_t1_metrics, print_t1_report


# ============================================================
# Callback: In sample prediction sau mỗi epoch
# ============================================================
class SamplePredictionCallback(TrainerCallback):
    """Sau mỗi epoch, lấy 1 sample ngẫu nhiên từ val set để dự đoán và in kết quả."""

    def __init__(self, val_json_path: str, tokenizer, max_length: int):
        with open(val_json_path, "r", encoding="utf-8") as f:
            self.val_samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        sample = random.choice(self.val_samples)
        true_label = sample["label"]
        text = sample["text_segmented"]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = T1_LABELS[pred_idx]

        epoch = int(state.epoch)
        check = "OK" if pred_label == true_label else "SAI"
        print(f"\n  [Epoch {epoch} Sample] Text: {text[:80]}...")
        print(f"    True: {true_label} | Pred: {pred_label} ({probs[pred_idx]:.2%}) [{check}]")


def train_t1(cfg: TrainingConfig):
    """Huấn luyện mô hình T1."""

    print("=" * 60)
    print("T1 – SCAM DETECTION TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------------
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------
    train_dataset = ScamT1Dataset(
        os.path.join(cfg.processed_data_dir, "t1_train.json"),
        tokenizer,
        cfg.max_seq_length,
    )
    val_dataset = ScamT1Dataset(
        os.path.join(cfg.processed_data_dir, "t1_val.json"),
        tokenizer,
        cfg.max_seq_length,
    )
    test_dataset = ScamT1Dataset(
        os.path.join(cfg.processed_data_dir, "t1_test.json"),
        tokenizer,
        cfg.max_seq_length,
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # --------------------------------------------------------
    # 3. Load model – fully fine-tuning (mục 8.3)
    # --------------------------------------------------------
    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(T1_LABELS),
        id2label=T1_ID2LABEL,
        label2id=T1_LABEL2ID,
    )
    # Không freeze encoder – fully fine-tuning
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --------------------------------------------------------
    # 4. Training arguments (mục 9)
    # --------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir_t1,
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
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        logging_dir=os.path.join(cfg.output_dir_t1, "logs"),
        logging_steps=10,
        report_to="none",
        fp16=False,  # Bật nếu GPU hỗ trợ
    )

    # --------------------------------------------------------
    # 5. Trainer
    # --------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_t1_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            SamplePredictionCallback(
                val_json_path=os.path.join(cfg.processed_data_dir, "t1_val.json"),
                tokenizer=tokenizer,
                max_length=cfg.max_seq_length,
            ),
        ],
    )

    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------
    print("\nBắt đầu huấn luyện T1...")
    train_result = trainer.train()
    print(f"\nTraining hoàn tất! Loss cuối: {train_result.training_loss:.4f}")

    # Lưu model tốt nhất
    trainer.save_model(os.path.join(cfg.output_dir_t1, "best_model"))
    tokenizer.save_pretrained(os.path.join(cfg.output_dir_t1, "best_model"))

    # --------------------------------------------------------
    # 7. Evaluation trên validation
    # --------------------------------------------------------
    print("\nĐánh giá trên validation set:")
    val_metrics = trainer.evaluate(val_dataset)
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # --------------------------------------------------------
    # 8. Evaluation trên test set (mục 10)
    # --------------------------------------------------------
    print("\nĐánh giá trên test set:")
    test_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids

    # In report chi tiết
    print_t1_report(test_labels, test_preds)

    # In metrics
    test_metrics = test_output.metrics
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\nModel T1 đã lưu tại: {os.path.join(cfg.output_dir_t1, 'best_model')}")
    return trainer


# ============================================================
if __name__ == "__main__":
    cfg = TrainingConfig()
    # Override batch size nhỏ hơn nếu GPU yếu
    if "--small" in sys.argv:
        cfg.per_device_train_batch_size = 8
        cfg.per_device_eval_batch_size = 16
        cfg.num_train_epochs = 10
    train_t1(cfg)
