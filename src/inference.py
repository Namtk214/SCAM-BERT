"""
Module inference cho T1 và T4.

Cho phép load model đã train và dự đoán trên input mới.
Input mới sẽ được qua pipeline: clean → segment → tokenize → predict.
"""

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import T1_LABELS, TACTIC_LABELS, TrainingConfig
from preprocessing import WordSegmenter, clean_text, neutralize_speakers


class ScamDetector:
    """
    Inference wrapper cho cả T1 và T4.
    """

    def __init__(
        self,
        t1_model_path: str,
        t4_model_path: str,
        vncorenlp_dir: str,
        t4_threshold: float = 0.5,
        max_length: int = 256,
    ):
        self.max_length = max_length
        self.t4_threshold = t4_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load word segmenter
        self.segmenter = WordSegmenter(vncorenlp_dir)

        # Load T1 model
        self.t1_tokenizer = AutoTokenizer.from_pretrained(t1_model_path)
        self.t1_model = AutoModelForSequenceClassification.from_pretrained(t1_model_path)
        self.t1_model.to(self.device)
        self.t1_model.eval()

        # Load T4 model
        self.t4_tokenizer = AutoTokenizer.from_pretrained(t4_model_path)
        self.t4_model = AutoModelForSequenceClassification.from_pretrained(t4_model_path)
        self.t4_model.to(self.device)
        self.t4_model.eval()

    def predict_t1(self, messages: list) -> dict:
        """
        Dự đoán T1 cho một cuộc hội thoại.

        Parameters
        ----------
        messages : list of dict
            Mỗi dict có keys: speaker_role, text

        Returns
        -------
        dict với label, confidence, và probabilities cho mỗi class
        """
        # Neutralize speakers
        neutral_messages = neutralize_speakers(messages)
        parts = []
        for msg in neutral_messages:
            text = clean_text(msg["text"])
            parts.append(f'{msg["speaker_neutral"]}: {text}')
        text_clean = " ".join(parts)

        # Word segmentation
        text_segmented = self.segmenter.segment(text_clean)

        # Tokenize
        inputs = self.t1_tokenizer(
            text_segmented,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.t1_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)

        return {
            "label": T1_LABELS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                label: float(probs[i]) for i, label in enumerate(T1_LABELS)
            },
        }

    def predict_t4(self, text: str) -> dict:
        """
        Dự đoán T4 cho một câu thoại của scammer.

        Parameters
        ----------
        text : str
            Câu thoại raw (chưa segment).

        Returns
        -------
        dict với predicted_tactics, probabilities cho mỗi tactic
        """
        text_clean = clean_text(text)
        text_segmented = self.segmenter.segment(text_clean)

        inputs = self.t4_tokenizer(
            text_segmented,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.t4_model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predicted_tactics = []
        for i, label in enumerate(TACTIC_LABELS):
            if probs[i] >= self.t4_threshold:
                predicted_tactics.append(label)

        return {
            "predicted_tactics": predicted_tactics,
            "probabilities": {
                label: float(probs[i]) for i, label in enumerate(TACTIC_LABELS)
            },
        }

    def analyze_conversation(self, messages: list) -> dict:
        """
        Phân tích toàn diện một cuộc hội thoại:
        - T1: phân loại toàn bộ conversation.
        - T4: phân loại tactic cho từng turn (nếu là SCAM/AMBIGUOUS).
        """
        # T1 prediction
        t1_result = self.predict_t1(messages)

        # T4 predictions cho từng turn
        turn_analysis = []
        for msg in messages:
            turn_info = {
                "speaker_role": msg["speaker_role"],
                "text": msg["text"],
            }
            # Chỉ phân tích tactic nếu conversation có dấu hiệu scam
            if t1_result["label"] in ("SCAM", "AMBIGUOUS"):
                t4_result = self.predict_t4(msg["text"])
                turn_info["tactics"] = t4_result
            turn_analysis.append(turn_info)

        return {
            "conversation_level": t1_result,
            "turn_level_analysis": turn_analysis,
        }


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    cfg = TrainingConfig()

    # Ví dụ sử dụng
    print("Demo inference (cần model đã train):")
    print()
    print("Ví dụ code:")
    print("""
    detector = ScamDetector(
        t1_model_path="outputs/t1_scam_detection/best_model",
        t4_model_path="outputs/t4_tactic_classification/best_model",
        vncorenlp_dir="vncorenlp",
        t4_threshold=0.45,  # threshold đã tune
    )

    messages = [
        {"speaker_role": "normal", "text": "Alo ai đấy?"},
        {"speaker_role": "scammer", "text": "Tôi là công an, bạn đang bị điều tra!"},
    ]

    result = detector.analyze_conversation(messages)
    print(result)
    """)
