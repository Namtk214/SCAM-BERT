"""
Entry point chính – chạy toàn bộ pipeline:
  1. Preprocessing (raw → clean → segment → chia tập)
  2. Train T1 (Scam Detection)
  3. Train T4 (Tactic Classification)

Usage:
  python run_pipeline.py                    # Chạy tất cả
  python run_pipeline.py --preprocess       # Chỉ preprocessing
  python run_pipeline.py --train-t1         # Chỉ train T1
  python run_pipeline.py --train-t4         # Chỉ train T4
  python run_pipeline.py --small            # Batch size nhỏ (GPU yếu)
"""

import argparse
import sys
import os

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="PhoBERT Fine-tuning Pipeline cho Scam Dataset"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Chỉ chạy preprocessing"
    )
    parser.add_argument(
        "--train-t1", action="store_true", help="Chỉ train T1 (Scam Detection)"
    )
    parser.add_argument(
        "--train-t4", action="store_true", help="Chỉ train T4 (Tactic Classification)"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Dùng batch size nhỏ và ít epoch hơn (cho GPU yếu / debug)"
    )
    parser.add_argument(
        "--model", type=str, default="vinai/phobert-base",
        help="Checkpoint PhoBERT (default: vinai/phobert-base)"
    )
    args = parser.parse_args()

    # Config
    cfg = TrainingConfig()
    cfg.model_name = args.model

    if args.small:
        cfg.per_device_train_batch_size = 8
        cfg.per_device_eval_batch_size = 16
        cfg.num_train_epochs = 10

    # Nếu không chọn gì → chạy tất cả
    run_all = not (args.preprocess or args.train_t1 or args.train_t4)

    # Step 1: Preprocessing
    if run_all or args.preprocess:
        from preprocessing import run_preprocessing
        run_preprocessing(cfg)

    # Step 2: Train T1
    if run_all or args.train_t1:
        from train_t1 import train_t1
        train_t1(cfg)

    # Step 3: Train T4
    if run_all or args.train_t4:
        from train_t4 import train_t4
        train_t4(cfg)

    print("\n" + "=" * 60)
    print("PIPELINE HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
