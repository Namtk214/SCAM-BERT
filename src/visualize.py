"""
Visualize dataset: phân bố nhãn, chiều dài, co-occurrence, ...

Chạy độc lập:
  python src/visualize.py --data-path data/raw_conversations.json

Hoặc qua pipeline:
  python run_pipeline.py --data-path data/raw_conversations.json --visualize
"""

import json
import os
import argparse
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import T1_LABELS, TACTIC_LABELS


# ============================================================
# Màu sắc
# ============================================================
COLORS_T1 = {"SCAM": "#e74c3c", "AMBIGUOUS": "#f39c12", "LEGIT": "#2ecc71"}
COLORS_T4 = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"]


def load_conversations(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 1. Phan bo nhan T1
# ============================================================
def plot_t1_distribution(conversations: list, save_dir: str):
    """Bar chart phân bố SCAM / AMBIGUOUS / LEGIT."""
    labels = [c["t1_label"] for c in conversations]
    counts = Counter(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        T1_LABELS,
        [counts.get(l, 0) for l in T1_LABELS],
        color=[COLORS_T1[l] for l in T1_LABELS],
        edgecolor="white",
        linewidth=1.5,
    )

    # Ghi số lên mỗi cột
    for bar, label in zip(bars, T1_LABELS):
        count = counts.get(label, 0)
        pct = count / len(labels) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{count} ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_title("T1 - Phan bo nhan Scam Detection", fontsize=14, fontweight="bold")
    ax.set_ylabel("So luong conversation")
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "t1_label_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: t1_label_distribution.png")


# ============================================================
# 2. Phan bo nhan T4 (tactic)
# ============================================================
def plot_t4_distribution(conversations: list, save_dir: str):
    """Bar chart tần suất từng tactic label."""
    tactic_counts = Counter()
    total_turns = 0

    for conv in conversations:
        for msg in conv["messages"]:
            if "t4_labels" in msg and msg["t4_labels"]:
                total_turns += 1
                for label in msg["t4_labels"]:
                    tactic_counts[label] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        TACTIC_LABELS,
        [tactic_counts.get(l, 0) for l in TACTIC_LABELS],
        color=COLORS_T4[:len(TACTIC_LABELS)],
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, label in zip(bars, TACTIC_LABELS):
        count = tactic_counts.get(label, 0)
        pct = count / total_turns * 100 if total_turns > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{count} ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title(
        f"T4 - Phan bo Tactic Labels ({total_turns} scammer turns)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("So lan xuat hien")
    ax.set_ylim(0, max(tactic_counts.values(), default=1) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "t4_tactic_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: t4_tactic_distribution.png")


# ============================================================
# 3. Co-occurrence matrix (T4)
# ============================================================
def plot_t4_cooccurrence(conversations: list, save_dir: str):
    """Heatmap: tactic nào hay đi cùng tactic nào."""
    n = len(TACTIC_LABELS)
    matrix = np.zeros((n, n), dtype=int)

    for conv in conversations:
        for msg in conv["messages"]:
            if "t4_labels" in msg and msg["t4_labels"]:
                indices = [TACTIC_LABELS.index(l) for l in msg["t4_labels"] if l in TACTIC_LABELS]
                # Diagonal: self count
                for idx in indices:
                    matrix[idx][idx] += 1
                # Off-diagonal: co-occurrence
                for i, j in combinations(indices, 2):
                    matrix[i][j] += 1
                    matrix[j][i] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(TACTIC_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(TACTIC_LABELS)

    # Ghi số vào ô
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_title("T4 - Tactic Co-occurrence Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="So lan xuat hien cung nhau")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "t4_cooccurrence.png"), dpi=150)
    plt.close()
    print(f"  Saved: t4_cooccurrence.png")


# ============================================================
# 4. So luong turn per conversation
# ============================================================
def plot_turn_distribution(conversations: list, save_dir: str):
    """Histogram: bao nhieu turn moi conversation."""
    turn_counts = [len(c["messages"]) for c in conversations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(turn_counts, bins=range(1, max(turn_counts) + 2),
            color="#3498db", edgecolor="white", linewidth=1.5, align="left")

    ax.set_title("Phan bo so turn moi conversation", fontsize=14, fontweight="bold")
    ax.set_xlabel("So turn")
    ax.set_ylabel("So conversation")
    ax.set_xticks(range(1, max(turn_counts) + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Stats
    avg = np.mean(turn_counts)
    ax.axvline(avg, color="#e74c3c", linestyle="--", linewidth=2, label=f"TB = {avg:.1f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "turn_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: turn_distribution.png")


# ============================================================
# 5. So tactic per scammer turn
# ============================================================
def plot_tactics_per_turn(conversations: list, save_dir: str):
    """Histogram: mỗi turn scammer có bao nhiêu tactic."""
    tactic_counts_per_turn = []
    for conv in conversations:
        for msg in conv["messages"]:
            if "t4_labels" in msg and msg["t4_labels"]:
                tactic_counts_per_turn.append(len(msg["t4_labels"]))

    if not tactic_counts_per_turn:
        print("  Skip: khong co tactic labels")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    max_count = max(tactic_counts_per_turn)
    ax.hist(tactic_counts_per_turn, bins=range(1, max_count + 2),
            color="#9b59b6", edgecolor="white", linewidth=1.5, align="left")

    ax.set_title("So tactic moi scammer turn", fontsize=14, fontweight="bold")
    ax.set_xlabel("So tactic")
    ax.set_ylabel("So turn")
    ax.set_xticks(range(1, max_count + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    avg = np.mean(tactic_counts_per_turn)
    ax.axvline(avg, color="#e74c3c", linestyle="--", linewidth=2, label=f"TB = {avg:.1f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tactics_per_turn.png"), dpi=150)
    plt.close()
    print(f"  Saved: tactics_per_turn.png")


# ============================================================
# 6. T1 label vs Turn count
# ============================================================
def plot_t1_vs_turns(conversations: list, save_dir: str):
    """Box plot: so turn theo tung T1 label."""
    data = {label: [] for label in T1_LABELS}
    for conv in conversations:
        data[conv["t1_label"]].append(len(conv["messages"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [data[l] for l in T1_LABELS]
    bp = ax.boxplot(box_data, labels=T1_LABELS, patch_artist=True, widths=0.5)

    for patch, label in zip(bp["boxes"], T1_LABELS):
        patch.set_facecolor(COLORS_T1[label])
        patch.set_alpha(0.7)

    ax.set_title("So turn theo T1 Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("So turn")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "t1_vs_turns.png"), dpi=150)
    plt.close()
    print(f"  Saved: t1_vs_turns.png")


# ============================================================
# 7. Summary stats
# ============================================================
def print_summary(conversations: list):
    """In thong ke tong quat."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Tong so conversation: {len(conversations)}")

    # T1
    t1_counts = Counter(c["t1_label"] for c in conversations)
    print(f"\n  T1 Labels:")
    for label in T1_LABELS:
        cnt = t1_counts.get(label, 0)
        pct = cnt / len(conversations) * 100
        print(f"    {label:12s}: {cnt:3d} ({pct:.1f}%)")

    # Turns
    turn_counts = [len(c["messages"]) for c in conversations]
    print(f"\n  Turns:")
    print(f"    Tong:    {sum(turn_counts)}")
    print(f"    TB:      {np.mean(turn_counts):.1f}")
    print(f"    Min/Max: {min(turn_counts)}/{max(turn_counts)}")

    # T4
    all_tactics = []
    scammer_turns = 0
    for conv in conversations:
        for msg in conv["messages"]:
            if "t4_labels" in msg and msg["t4_labels"]:
                scammer_turns += 1
                all_tactics.extend(msg["t4_labels"])

    tactic_counts = Counter(all_tactics)
    print(f"\n  T4 Scammer turns: {scammer_turns}")
    print(f"  T4 Tactic Labels:")
    for label in TACTIC_LABELS:
        cnt = tactic_counts.get(label, 0)
        pct = cnt / scammer_turns * 100 if scammer_turns > 0 else 0
        print(f"    {label:12s}: {cnt:3d} ({pct:.1f}%)")

    print()


# ============================================================
# Main
# ============================================================
def run_visualization(data_path: str, save_dir: str = None):
    """Chay toan bo visualization."""
    print("=" * 50)
    print("DATA VISUALIZATION")
    print("=" * 50)

    conversations = load_conversations(data_path)

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(data_path), "figures")
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Save dir: {save_dir}\n")

    # Summary
    print_summary(conversations)

    # Plots
    plot_t1_distribution(conversations, save_dir)
    plot_t4_distribution(conversations, save_dir)
    plot_t4_cooccurrence(conversations, save_dir)
    plot_turn_distribution(conversations, save_dir)
    plot_tactics_per_turn(conversations, save_dir)
    plot_t1_vs_turns(conversations, save_dir)

    print(f"\nTat ca bieu do da luu tai: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize scam dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw_conversations.json")
    parser.add_argument("--save-dir", type=str, default=None, help="Output folder for figures")
    args = parser.parse_args()
    run_visualization(args.data_path, args.save_dir)
