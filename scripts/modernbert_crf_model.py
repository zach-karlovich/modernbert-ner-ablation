"""
Wraps emissions + constrained CRF in a single module. Specifically for sentence-level
context. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForTokenClassification

from conll2003_labels import id2label, label2id, label_list
from crf_bio import make_constrained_crf


class ModernBertTokenCRF(nn.Module):
    def __init__(
        self,
        model_id: str,
        *,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=trust_remote_code,
        )
        self.crf = make_constrained_crf()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        emissions = outputs.logits
        mask = attention_mask.bool()
        if labels is not None:
            llh = self.crf(emissions, labels, mask=mask, reduction="sum")
            loss = -llh / mask.sum().clamp(min=1)
            return loss, emissions
        return None, emissions

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
        with torch.no_grad():
            _, emissions = self.forward(input_ids, attention_mask, labels=None)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


def build_crf_optimizer(
    model: ModernBertTokenCRF,
    lr: float,
    crf_lr: float,
    weight_decay: float = 0.01,
) -> optim.AdamW:
    no_decay = ["bias", "norm.weight"]
    decay: list[torch.nn.Parameter] = []
    no_decay_ps: list[torch.nn.Parameter] = []
    for n, p in model.base.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_ps.append(p)
        else:
            decay.append(p)
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_ps, "weight_decay": 0.0, "lr": lr},
            {"params": model.crf.parameters(), "weight_decay": 0.0, "lr": crf_lr},
        ]
    )
