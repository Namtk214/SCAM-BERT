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
python run_pipeline.py --data-path /path/to/raw_conversations.json

# Hoặc dùng batch nhỏ hơn (GPU yếu / debug)
python run_pipeline.py --data-path /path/to/raw_conversations.json --small
```

### Hoặc chạy từng bước riêng

```bash
# Bước 2a: Chỉ preprocessing
python run_pipeline.py --data-path /path/to/raw_conversations.json --preprocess

# Bước 2b: Chỉ train T1 (Scam Detection)
python run_pipeline.py --train-t1

# Bước 2c: Chỉ train T4 (Tactic Classification)
python run_pipeline.py --train-t4
```

### Tuỳ chỉnh đường dẫn và model

```bash
# Chỉ định thư mục output
python run_pipeline.py --data-path /path/to/data.json --output-dir /path/to/output

# Dùng PhoBERT large
python run_pipeline.py --data-path /path/to/data.json --model vinai/phobert-large

# Kết hợp tất cả
python run_pipeline.py --data-path /path/to/data.json --output-dir ./results --model vinai/phobert-large --small
```

### Chạy chức năng Trực quan hóa dữ liệu (Visualize)

Pipeline hỗ trợ trích xuất thống kê và vẽ biểu đồ về các đặc trưng của dataset (phân bố nhãn, số lượng tactic, số turn...). 
Các biểu đồ sẽ được lưu dưới dạng file `.png` vào thư mục `figures/` bên trong đườg dẫn xuất dữ liệu (`output-dir`).

```bash
# Chỉ chạy visualize (không train)
python run_pipeline.py --data-path /path/to/raw_conversations.json --visualize

# Kết hợp preprocess + train nhỏ + visualize
python run_pipeline.py --data-path /path/to/raw_conversations.json --visualize --small
```

Danh sách các biểu đồ sinh ra:
1. `t1_label_distribution.png`: Phân bố SCAM / AMBIGUOUS / LEGIT.
2. `t4_tactic_distribution.png`: Tần suất xuất hiện của Tactic (T4).
3. `t4_cooccurrence.png`: Heatmap về việc nhãn T4 nào hay đi cùng nhau.
4. `turn_distribution.png`: Histogram số turn trên mỗi conversation.
5. `tactics_per_turn.png`: Histogram số tactic trung bình / scammer turn.
6. `t1_vs_turns.png`: So sánh độ rộng cực tiểu theo độ dài turn.

---

## Flag dòng lệnh

| Flag | Mô tả |
|------|--------|
| `--data-path <path>` | **(Bắt buộc khi preprocess)** Đường dẫn tới file `raw_conversations.json` |
| `--output-dir <path>` | Thư mục lưu model và processed data (mặc định: `./outputs`) |
| `--preprocess` | Chỉ chạy preprocessing |
| `--visualize` | Trực quan hóa thống kê dataset (lưu biểu đồ vào thư mục `figures/`) |
| `--train-t1` | Chỉ train T1 (Scam Detection) |
| `--train-t4` | Chỉ train T4 (Tactic Classification) |
| `--small` | Batch size nhỏ (8) + 10 epoch (debug / GPU yếu) |
| `--model <name>` | Checkpoint PhoBERT (mặc định: `vinai/phobert-base`) |

Không truyền `--preprocess`, `--train-t1`, `--train-t4` → chạy tất cả.

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
