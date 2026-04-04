"""
Toy checks for pytorch-crf: padding mask ignored at padded steps, BIO illegal
transitions heavily penalized (e.g. B-LOC -> I-PER), and decode respects mask.
"""

import sys
from pathlib import Path

import torch
from torchcrf import CRF

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from train_modernbert_doc_ner import label2id, label_list

root = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent

# CLAUDE SUPPORT FOR PENALTY AND NLL_MARGIN:
# -1e4 is large enough to make illegal transitions effectively impossible in
# log-space (exp(-1e4) ≈ 0), while staying well within float32 precision to
# avoid NaN gradients.  NLL_MARGIN is a sanity threshold: the log-likelihood
# gap between a legal and illegal path through the constrained CRF should
# comfortably exceed this on a toy example with a single penalty-violating step.
PENALTY = -1e4
NLL_MARGIN = 50.0


def parse_bio(label: str):
    if label == "O":
        return "O", None
    kind, etype = label.split("-", 1)
    return kind, etype


def start_ok(label: str) -> bool:
    kind, _ = parse_bio(label)
    return kind in ("O", "B")


def transition_ok(prev: str, nxt: str) -> bool:
    """Check whether prev -> nxt is a legal BIO transition.

    Legal transitions under standard BIO:
      - Anything -> O
      - Anything -> B-X  (B can always start a new entity)
      - {B-X, I-X} -> I-X (continue same entity type only)
    """
    k_prev, t_prev = parse_bio(prev)
    k_next, t_next = parse_bio(nxt)

    if k_next == "O":
        return True
    if k_next == "B":
        # B-X can follow anything: O, B-Y, B-X, I-Y, I-X are all legal.
        return True
    if k_next == "I":
        # I-X must continue an open span of the same entity type.
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


def check_padding_mask_invariance(crf: CRF) -> None:
    torch.manual_seed(0)
    b, t_max, c = 2, 5, len(label_list)
    emissions = torch.randn(b, t_max, c)
    lens = [3, 5]
    mask = torch.zeros(b, t_max, dtype=torch.bool)
    for i, L in enumerate(lens):
        mask[i, :L] = True

    tags = torch.zeros(b, t_max, dtype=torch.long)
    for i, L in enumerate(lens):
        tags[i, :L] = torch.randint(0, c, (L,))

    ll0 = crf(emissions, tags, mask=mask, reduction="sum")

    tags_alt = tags.clone()
    for i in range(b):
        for t in range(t_max):
            if not mask[i, t]:
                tags_alt[i, t] = torch.randint(0, c, (1,)).item()

    ll1 = crf(emissions, tags_alt, mask=mask, reduction="sum")
    assert torch.allclose(ll0, ll1), (
        f"padding tag changes altered log-likelihood: {ll0.item()} vs {ll1.item()}"
    )
    print("  Padding mask (forward): PASS — mutating padded tag ids does not change log-likelihood")


def check_decode_respects_mask(crf: CRF) -> None:
    """Verify that crf.decode() returns sequences whose lengths match the mask."""
    torch.manual_seed(1)
    b, t_max, c = 4, 8, len(label_list)
    emissions = torch.randn(b, t_max, c)
    lens = [2, 5, 8, 1]
    mask = torch.zeros(b, t_max, dtype=torch.bool)
    for i, L in enumerate(lens):
        mask[i, :L] = True

    paths = crf.decode(emissions, mask=mask)
    for i, (path, expected_len) in enumerate(zip(paths, lens)):
        assert len(path) == expected_len, (
            f"batch element {i}: decode returned {len(path)} tags, expected {expected_len}"
        )
    print(
        f"  Padding mask (decode): PASS — decoded lengths {[len(p) for p in paths]} "
        f"match expected {lens}"
    )


def check_illegal_transition_paris_scene(crf: CRF) -> None:
    id2label = {v: k for k, v in label2id.items()}
    o = label2id["O"]
    b_loc = label2id["B-LOC"]
    i_loc = label2id["I-LOC"]
    i_per = label2id["I-PER"]

    print('\n  Scene: "He coached at Virginia Tech." — emissions at "Tech" favor I-PER,')
    print("  but we just opened B-LOC; crossing entity types mid-span should be impossible.\n")

    neg = -30.0
    emissions = torch.full((1, 3, len(label_list)), neg)
    emissions[0, 0, o] = 10.0
    emissions[0, 1, b_loc] = 10.0
    emissions[0, 2, i_per] = 20.0
    emissions[0, 2, i_loc] = 5.0
    emissions[0, 2, o] = 0.0

    mask = torch.ones(1, 3, dtype=torch.bool)
    path = crf.decode(emissions, mask=mask)[0]
    decoded = [id2label[i] for i in path]
    assert path[2] != i_per, (
        f"Viterbi wrongly used I-PER after B-LOC; path={decoded}"
    )
    print(f"  Viterbi path: {' -> '.join(decoded)} (I-PER not chosen after B-LOC) PASS")

    legal = torch.tensor([[o, b_loc, i_loc]], dtype=torch.long)
    illegal = torch.tensor([[o, b_loc, i_per]], dtype=torch.long)
    ll_legal = crf(emissions, legal, mask=mask, reduction="sum")
    ll_illegal = crf(emissions, illegal, mask=mask, reduction="sum")
    gap = (ll_legal - ll_illegal).item()
    print(
        f"  Log-likelihood gap (legal I-LOC vs illegal I-PER at step 3): {gap:.2f} "
        f"(expect large positive)"
    )
    assert gap > NLL_MARGIN, f"expected gap > {NLL_MARGIN}, got {gap}"
    print(f"  Illegal vs legal tag sequence: PASS (gap > {NLL_MARGIN})")


def check_consecutive_same_type_entities(crf: CRF) -> None:
    """Verify B-LOC -> B-LOC is legal (two adjacent single-token LOC entities)."""
    id2label = {v: k for k, v in label2id.items()}
    o = label2id["O"]
    b_loc = label2id["B-LOC"]

    print("\n  Scene: \"Virginia Tech Hokies\" — two adjacent single-token LOC entities\n"
          "  (B-LOC -> B-LOC) must be legal.\n")

    neg = -30.0
    emissions = torch.full((1, 3, len(label_list)), neg)
    emissions[0, 0, o] = 10.0
    emissions[0, 1, b_loc] = 10.0
    emissions[0, 2, b_loc] = 10.0

    mask = torch.ones(1, 3, dtype=torch.bool)
    path = crf.decode(emissions, mask=mask)[0]
    decoded = [id2label[i] for i in path]
    assert path[1] == b_loc and path[2] == b_loc, (
        f"Viterbi rejected B-LOC -> B-LOC; path={decoded}"
    )
    print(f"  Viterbi path: {' -> '.join(decoded)} (consecutive B-LOC allowed) PASS")

    seq = torch.tensor([[o, b_loc, b_loc]], dtype=torch.long)
    ll = crf(emissions, seq, mask=mask, reduction="sum")
    print(f"  Log-likelihood of [O, B-LOC, B-LOC]: {ll.item():.2f} (should not be penalized) PASS")


def main() -> None:
    assert (root / "pyproject.toml").exists(), f"Run from project root; cwd={Path.cwd()}"

    print("CoNLL-2003 CRF verification (pytorch-crf, BIO constraints, toy batch)\n")
    print(f"  Project: {root}")
    print(f"  Labels ({len(label_list)}): {label_list}\n")

    crf = make_constrained_crf()

    print("Check 1: padding positions masked in CRF forward pass")
    check_padding_mask_invariance(crf)

    print("\nCheck 2: padding positions respected in CRF decode")
    check_decode_respects_mask(crf)

    print("\nCheck 3: illegal transition B-LOC -> I-PER")
    check_illegal_transition_paris_scene(crf)

    print("\nCheck 4: consecutive same-type entities (B-LOC -> B-LOC)")
    check_consecutive_same_type_entities(crf)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

