"""
Utility để load PhoBERT model với fix cho LayerNorm key naming.

PhoBERT checkpoint (vinai/phobert-base) dùng naming cũ:
  LayerNorm.gamma → cần đổi thành LayerNorm.weight
  LayerNorm.beta  → cần đổi thành LayerNorm.bias

Nếu không fix, toàn bộ LayerNorm sẽ bị random init
và model gần như mất hết pretrained knowledge.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig


def _fix_state_dict_keys(state_dict: dict) -> dict:
    """Rename LayerNorm.gamma/beta → LayerNorm.weight/bias."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "LayerNorm.gamma" in key:
            new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
        elif "LayerNorm.beta" in key:
            new_key = key.replace("LayerNorm.beta", "LayerNorm.bias")
        new_state_dict[new_key] = value
    return new_state_dict


def load_phobert_for_classification(
    model_name: str,
    num_labels: int,
    id2label: dict = None,
    label2id: dict = None,
    problem_type: str = None,
) -> AutoModelForSequenceClassification:
    """
    Load PhoBERT với fix cho LayerNorm keys.

    Parameters
    ----------
    model_name : str
        Checkpoint name (e.g. "vinai/phobert-base")
    num_labels : int
        Số class output
    id2label, label2id : dict
        Label mappings
    problem_type : str
        "single_label_classification" hoặc "multi_label_classification"

    Returns
    -------
    AutoModelForSequenceClassification đã load đúng pretrained weights
    """
    # 1. Load config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    if problem_type:
        config.problem_type = problem_type

    # 2. Khởi tạo model trống với config
    model = AutoModelForSequenceClassification.from_config(config)

    # 3. Load pretrained state_dict và fix keys
    pretrained = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    state_dict = pretrained.state_dict()
    fixed_state_dict = _fix_state_dict_keys(state_dict)

    # 4. Load vào model, cho phép thiếu classifier head (sẽ random init)
    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)

    # Chỉ classifier head mới được phép missing (random init)
    real_missing = [k for k in missing if "classifier" not in k]
    if real_missing:
        print(f"  WARNING: Các key sau vẫn missing sau khi fix: {real_missing}")
    else:
        print(f"  PhoBERT LayerNorm keys fixed OK! ({len(fixed_state_dict)} params loaded)")

    del pretrained
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return model
