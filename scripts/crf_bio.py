"""
Constrained CRF for BIO-labeled NER data. Returns constrained CRF object
so illegal transitions are heavily penalized.
"""

import torch
from torchcrf import CRF

from conll2003_labels import label_list

PENALTY = -1e4


def parse_bio(label: str):
    if label == "O":
        return "O", None
    kind, etype = label.split("-", 1)
    return kind, etype


def start_ok(label: str) -> bool:
    kind, _ = parse_bio(label)
    return kind in ("O", "B")


def transition_ok(prev: str, nxt: str) -> bool:
    k_prev, t_prev = parse_bio(prev)
    k_next, t_next = parse_bio(nxt)

    if k_next == "O":
        return True
    if k_next == "B":
        return True
    if k_next == "I":
        return k_prev in ("B", "I") and t_prev == t_next
    return False


def apply_bio_constraints(crf: CRF, labels: list[str], penalty: float = PENALTY) -> None:
    n = len(labels)
    with torch.no_grad():
        crf.transitions.fill_(penalty)
        for i in range(n):
            for j in range(n):
                if transition_ok(labels[i], labels[j]):
                    crf.transitions[i, j] = 0.0

        crf.start_transitions.fill_(penalty)
        for j in range(n):
            if start_ok(labels[j]):
                crf.start_transitions[j] = 0.0

        crf.end_transitions.zero_()


def make_constrained_crf() -> CRF:
    crf = CRF(len(label_list), batch_first=True)
    apply_bio_constraints(crf, label_list)
    return crf
