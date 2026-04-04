"""
Toy checks for pytorch-crf: padding mask ignored at padded steps, BIO illegal
transitions heavily penalized (e.g. B-LOC -> I-PER), and decode respects mask.

Updated to use regression checks for:

- padding/masks
- transitions
- CRF dense labeling against dataset 
"""

import sys
from pathlib import Path

import torch

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from conll2003_labels import label2id, label_list
from crf_bio import make_constrained_crf
from dense_bio_labels import (
    assign_dense_bio_labels,
    collapse_to_word_labels,
    word_ids_list_to_tensor_ids,
)

root = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent

NLL_MARGIN = 50.0


def check_padding_mask_invariance(crf) -> None:
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


def check_decode_respects_mask(crf) -> None:
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


def check_illegal_transition_paris_scene(crf) -> None:
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


def check_consecutive_same_type_entities(crf) -> None:
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


def check_dense_label_roundtrip_manual() -> None:
    from transformers import AutoTokenizer

    id2label = {i: label for i, label in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    words = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb"]
    tags = ["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O"]
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )
    wids = enc.word_ids()
    dense_ids = assign_dense_bio_labels(wids, tags)
    row_w = word_ids_list_to_tensor_ids(wids)
    mask = enc["attention_mask"][0].tolist()
    collapsed = collapse_to_word_labels(row_w, dense_ids, mask, id2label)
    assert collapsed == tags, f"sentence roundtrip: {collapsed!r} vs {tags!r}"
    print("  Dense BIO roundtrip (manual sentence): PASS")


def check_dense_label_roundtrip_datasets() -> None:
    from transformers import AutoTokenizer

    from train_modernbert_crf_ner import ConllDatasetCRF, parse_conll
    from train_modernbert_doc_crf_ner import ConllDocContextDatasetCRF, parse_conll_documents

    id2label = {i: label for i, label in enumerate(label_list)}
    data_dir = root / "data" / "conll2003"
    if not (data_dir / "eng.train").exists():
        print("  Dense BIO roundtrip (datasets): SKIP — data/conll2003 not present")
        return

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    sents = parse_conll(data_dir / "eng.train")
    ds_s = ConllDatasetCRF(sents[:50], tok, max_length=512)
    for i in range(min(10, len(ds_s))):
        item = ds_s[i]
        L = int(item["attention_mask"].sum())
        row_w = item["word_ids"][:L].tolist()
        row_m = item["attention_mask"][:L].tolist()
        gold_dense = item["labels"][:L].tolist()
        collapsed = collapse_to_word_labels(row_w, gold_dense, row_m, id2label)
        gold_words = [t for _, t in sents[i]]
        assert collapsed == gold_words, f"sentence idx {i}: {collapsed!r} vs {gold_words!r}"

    docs = parse_conll_documents(data_dir / "eng.train")
    ds_d = ConllDocContextDatasetCRF(docs, tok, max_length=512)
    for i in range(min(10, len(ds_d))):
        item = ds_d[i]
        L = int(item["attention_mask"].sum())
        row_w = item["word_ids"][:L].tolist()
        row_m = item["attention_mask"][:L].tolist()
        gold_dense = item["labels"][:L].tolist()
        winc = item["word_include_mask"]
        collapsed = collapse_to_word_labels(
            row_w, gold_dense, row_m, id2label, word_include_mask=winc
        )
        d_idx, s_idx = ds_d.targets[i]
        target_sent = docs[d_idx][s_idx]
        gold_words = [t for _, t in target_sent]
        assert collapsed == gold_words, f"doc idx {i}: {collapsed!r} vs {gold_words!r}"

    print("  Dense BIO roundtrip (ConllDatasetCRF + ConllDocContextDatasetCRF): PASS")


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

    print("\nCheck 5: dense BIO assign + collapse roundtrip")
    check_dense_label_roundtrip_manual()
    check_dense_label_roundtrip_datasets()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
