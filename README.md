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

## Hướng dẫn chạy

### Bước 1: Cài dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ pipeline (1 lệnh duy nhất)

```bash
# Chạy tất cả: preprocessing → train T1 → train T4
python run_pipeline.py

# Hoặc dùng batch nhỏ hơn (GPU yếu / debug)
python run_pipeline.py --small
```

### Hoặc chạy từng bước riêng

```bash
# Bước 2a: Chỉ preprocessing (tải VnCoreNLP + word segment + chia tập)
python run_pipeline.py --preprocess

# Bước 2b: Chỉ train T1 (Scam Detection)
python run_pipeline.py --train-t1

# Bước 2c: Chỉ train T4 (Tactic Classification)
python run_pipeline.py --train-t4
```

### Bước 3: Dùng model khác (tuỳ chọn)

```bash
# Dùng PhoBERT large thay vì base
python run_pipeline.py --model vinai/phobert-large

# Kết hợp
python run_pipeline.py --model vinai/phobert-large --small
```

---

## Flag dòng lệnh

| Flag | Mô tả |
|------|--------|
| `--preprocess` | Chỉ chạy preprocessing |
| `--train-t1` | Chỉ train T1 (Scam Detection) |
| `--train-t4` | Chỉ train T4 (Tactic Classification) |
| `--small` | Batch size nhỏ (8) + 10 epoch (debug / GPU yếu) |
| `--model <name>` | Checkpoint PhoBERT (mặc định: `vinai/phobert-base`) |

Không truyền flag nào → chạy tất cả (preprocessing + T1 + T4).

---

## Cấu hình

Tất cả hyperparameters định nghĩa trong `src/config.py`:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `model_name` | `vinai/phobert-base` | Checkpoint PhoBERT |
| `max_seq_length` | 256 | Độ dài tối đa input (theo paper) |
| `learning_rate` | 1e-5 | Learning rate (theo paper) |
| `num_train_epochs` | 30 | Số epoch |
| `per_device_train_batch_size` | 32 | Batch size |
| `val_ratio` | 0.15 | Tỷ lệ validation |
| `test_ratio` | 0.15 | Tỷ lệ test |
| `t4_threshold` | 0.5 | Ngưỡng sigmoid mặc định cho T4 |

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
