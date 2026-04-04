"""
Dense BIO label assignment for dense NER data. Turns word-level BIO tags
into dense BIO labels, e.g. "B-ORG I-ORG I-ORG" -> [1, 1, 1]. Collapses to
one label per word for seqeval comparisons.
"""

from conll2003_labels import label2id


def continuation_tag(word_tag: str) -> str:
    if word_tag == "O":
        return "O"
    _, etype = word_tag.split("-", 1)
    return f"I-{etype}"


def assign_dense_bio_labels(word_ids: list, tags: list[str]) -> list[int]:
    out: list[int] = []
    for i, wid in enumerate(word_ids):
        if wid is None:
            out.append(label2id["O"])
            continue
        if i == 0 or word_ids[i - 1] != wid:
            out.append(label2id[tags[wid]])
        else:
            out.append(label2id[continuation_tag(tags[wid])])
    return out


def collapse_to_word_labels(
    row_word_ids: list[int],
    row_token_label_ids: list[int],
    row_attention_mask: list[int] | list[bool],
    id2label: dict[int, str],
    word_include_mask: list[bool] | None = None,
) -> list[str]:
    seq: list[str] = []
    valid_words = [w for w in row_word_ids if w >= 0]
    n_words = max(valid_words, default=-1) + 1
    if word_include_mask is None:
        word_include_mask = [True] * n_words
    for j in range(len(row_word_ids)):
        if not row_attention_mask[j]:
            continue
        wid = row_word_ids[j]
        if wid < 0:
            continue
        if j > 0 and row_word_ids[j - 1] == wid:
            continue
        if wid < len(word_include_mask) and not word_include_mask[wid]:
            continue
        seq.append(id2label[int(row_token_label_ids[j])])
    return seq


def word_ids_list_to_tensor_ids(word_ids: list) -> list[int]:
    return [-1 if w is None else w for w in word_ids]
