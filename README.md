# PhoBERT Fine-tuning cho Scam Detection

Pipeline fine-tuning PhoBERT cho 2 task:
- **T1 – Scam Detection**: Phân loại cuộc hội thoại → `SCAM` / `AMBIGUOUS` / `LEGIT`
- **T4 – Tactic Classification**: Phân loại tactic tâm lý scammer (multi-label)

---

## Cấu trúc dự án

```
Bert/
├── run_pipeline.py                 # Entry point chính
├── requirements.txt                # Dependencies
├── README.md                       # Hướng dẫn này
├── data/
│   └── raw_conversations.json      # 30 mẫu dataset gốc
└── src/
    ├── config.py                   # Cấu hình (taxonomy, hyperparams)
    ├── preprocessing.py            # Pipeline tiền xử lý + word segmentation
    ├── dataset.py                  # PyTorch Dataset classes (T1 & T4)
    ├── metrics.py                  # Hàm đánh giá (F1, confusion matrix, ...)
    ├── train_t1.py                 # Training T1 (Scam Detection)
    ├── train_t4.py                 # Training T4 (Tactic Classification)
    └── inference.py                # Module inference/dự đoán
```

---

## Hướng dẫn chạy trên Google Colab

### Bước 1: Upload project lên Google Drive

Upload toàn bộ thư mục `Bert/` lên Google Drive, ví dụ vào đường dẫn:
```
My Drive/Bert/
```

### Bước 2: Tạo notebook mới trên Colab

Vào [Google Colab](https://colab.research.google.com/) → New Notebook.

**Quan trọng**: Chọn GPU Runtime:
- Menu → `Runtime` → `Change runtime type` → chọn **GPU** (T4 miễn phí là đủ)

### Bước 3: Mount Google Drive & cài dependencies

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn project trên Drive
PROJECT_DIR = "/content/drive/MyDrive/Bert"
```

```python
# Cell 2: Cài dependencies
!pip install -r {PROJECT_DIR}/requirements.txt
```

```python
# Cell 3: Thêm src vào path
import sys
sys.path.insert(0, f"{PROJECT_DIR}/src")
```

### Bước 4: Chạy Preprocessing

```python
# Cell 4: Preprocessing (tải VnCoreNLP + word segment + chia tập)
import os
os.chdir(PROJECT_DIR)

from config import TrainingConfig
from preprocessing import run_preprocessing

cfg = TrainingConfig()
run_preprocessing(cfg)
```

**Output kỳ vọng:**
```
============================================================
PREPROCESSING PIPELINE
============================================================
Đang tải VnCoreNLP models vào .../vncorenlp...
Đọc dữ liệu từ .../raw_conversations.json...
  Tổng số conversation: 30
  Train: 21 | Val: 4 | Test: 5
  T1 train: 21 samples → .../t1_train.json
  T4 train: XX samples → .../t4_train.json
  ...
Preprocessing hoàn tất!
```

### Bước 5: Train T1 (Scam Detection)

```python
# Cell 5: Train T1
from train_t1 import train_t1

cfg = TrainingConfig()
# Giảm batch size & epoch cho dataset nhỏ / debug nhanh
cfg.per_device_train_batch_size = 8
cfg.per_device_eval_batch_size = 16
cfg.num_train_epochs = 10

trainer_t1 = train_t1(cfg)
```

### Bước 6: Train T4 (Tactic Classification)

```python
# Cell 6: Train T4
from train_t4 import train_t4

cfg = TrainingConfig()
cfg.per_device_train_batch_size = 8
cfg.per_device_eval_batch_size = 16
cfg.num_train_epochs = 10

trainer_t4, best_threshold = train_t4(cfg)
print(f"Best threshold cho T4: {best_threshold:.2f}")
```

### Bước 7: Inference (thử dự đoán)

```python
# Cell 7: Test inference
from inference import ScamDetector

detector = ScamDetector(
    t1_model_path=f"{PROJECT_DIR}/outputs/t1_scam_detection/best_model",
    t4_model_path=f"{PROJECT_DIR}/outputs/t4_tactic_classification/best_model",
    vncorenlp_dir=f"{PROJECT_DIR}/vncorenlp",
    t4_threshold=best_threshold,
)

# Test với conversation mới
messages = [
    {"speaker_role": "normal", "text": "Alo ai đấy?"},
    {"speaker_role": "scammer", "text": "Tôi là công an quận 1, bạn đang bị điều tra vì rửa tiền!"},
    {"speaker_role": "normal", "text": "Hả? Tôi không biết gì hết."},
    {"speaker_role": "scammer", "text": "Chuyển ngay 50 triệu vào tài khoản tạm giữ, nếu không sẽ bị bắt!"},
]

result = detector.analyze_conversation(messages)

print(f"Kết quả T1: {result['conversation_level']['label']}")
print(f"Confidence: {result['conversation_level']['confidence']:.2%}")
print()
for turn in result['turn_level_analysis']:
    print(f"  [{turn['speaker_role']}]: {turn['text'][:60]}...")
    if 'tactics' in turn:
        print(f"    → Tactics: {turn['tactics']['predicted_tactics']}")
```

---

## Chạy nhanh 1 lệnh (toàn bộ pipeline)

Nếu muốn chạy tất cả trong 1 cell:

```python
# Chạy toàn bộ: preprocessing → train T1 → train T4
import os, sys
os.chdir("/content/drive/MyDrive/Bert")
sys.path.insert(0, "src")

!python run_pipeline.py --small
```

Flag `--small` sẽ dùng batch_size=8 và 10 epochs (phù hợp debug/dataset nhỏ).

---

## Cấu hình

Tất cả hyperparameters được định nghĩa trong `src/config.py`:

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|-------|
| `model_name` | `vinai/phobert-base` | Checkpoint PhoBERT |
| `max_seq_length` | 256 | Độ dài tối đa input (theo paper) |
| `learning_rate` | 1e-5 | Learning rate (theo paper) |
| `num_train_epochs` | 30 | Số epoch |
| `per_device_train_batch_size` | 32 | Batch size |
| `val_ratio` | 0.15 | Tỷ lệ validation |
| `test_ratio` | 0.15 | Tỷ lệ test |
| `t4_threshold` | 0.5 | Ngưỡng sigmoid mặc định cho T4 |

Có thể dùng `vinai/phobert-large` nếu có đủ GPU:
```python
cfg.model_name = "vinai/phobert-large"
```

---

## Taxonomy nhãn

### T1 – Scam Detection (single-label, 3 class)
| Label | Mô tả |
|-------|--------|
| `SCAM` | Cuộc hội thoại lừa đảo |
| `AMBIGUOUS` | Mơ hồ, chưa đủ kết luận |
| `LEGIT` | Hợp pháp, bình thường |

### T4 – Tactic Classification (multi-label, 5 tactic)
| Label | Mô tả |
|-------|--------|
| `SA_AUTH` | Giả mạo thẩm quyền (công an, ngân hàng, ...) |
| `SA_THREAT` | Đe dọa (bắt giữ, phạt, ...) |
| `SA_URGENCY` | Tạo cảm giác gấp (hạn chót, ngay lập tức, ...) |
| `SA_REASSURE` | Trấn an (yên tâm, hoàn tiền, ...) |
| `SA_DEFLECT` | Lảng tránh / chuyển hướng câu hỏi |

---

## Metrics đánh giá

| Task | Metrics |
|------|---------|
| T1 | Accuracy, Macro-F1, F1/Precision/Recall per class, Confusion matrix |
| T4 | Micro-F1, Macro-F1, Per-label F1, Exact match ratio, Precision/Recall per label |

---

## Lưu ý quan trọng

1. **Word segmentation bắt buộc**: PhoBERT yêu cầu input đã qua word segmentation (VnCoreNLP/RDRSegmenter). Pipeline tự xử lý việc này.

2. **Chia tập theo conversation_id**: Tránh data leakage – cùng 1 conversation không xuất hiện ở cả train và test.

3. **Fully fine-tuning**: Toàn bộ tham số PhoBERT + classification head đều được cập nhật (không freeze encoder).

4. **Threshold tuning cho T4**: Mặc định 0.5 chỉ là điểm xuất phát. Pipeline tự động tune threshold trên validation set.

5. **Dataset hiện tại chỉ có 30 mẫu**: Đây là demo. Để có kết quả tốt, cần ít nhất vài trăm mẫu conversation.
