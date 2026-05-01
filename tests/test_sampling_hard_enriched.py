from __future__ import annotations

from obliqaxref.generate.common.sampling import sample_xref_rows


def _row(src_id: str, tgt_id: str, src_doc: str, tgt_doc: str, src_text: str, tgt_text: str, ref: str = ""):
    return {
        "SourceID": src_id,
        "TargetID": tgt_id,
        "SourceDocumentID": src_doc,
        "TargetDocumentID": tgt_doc,
        "SourcePassage": src_text,
        "TargetPassage": tgt_text,
        "ReferenceText": ref,
    }


def test_hard_enriched_returns_requested_and_deterministic():
    rows = [
        _row("S1","T1","D1","D2","alpha beta gamma", "delta epsilon zeta"),
        _row("S2","T2","D1","D1","alpha beta", "alpha beta"),  # high overlap, same doc
        _row("S3","T3","D2","D3","lorem ipsum dolor sit amet", "consectetur adipiscing elit"),
        _row("S4","T4","D4","D5","short", "short"),
        _row("S5","T5","D6","D7","terms conditions obligations requirement", "exceptions deadlines numeric 10%"),
    ]
    n = 3
    seed = 42

    sel1, rep1 = sample_xref_rows(rows, n=n, mode="hard_enriched", seed=seed)
    sel2, rep2 = sample_xref_rows(rows, n=n, mode="hard_enriched", seed=seed)

    assert len(sel1) == n
    assert len(sel2) == n
    # Deterministic with seed
    assert [ (r["SourceID"], r["TargetID"]) for r in sel1 ] == [ (r["SourceID"], r["TargetID"]) for r in sel2 ]
    assert rep1["sampling_mode"] == "hard_enriched"
    assert rep2["sampling_mode"] == "hard_enriched"


def test_hard_enriched_prefers_lower_overlap():
    # Construct rows where one pair has much lower overlap and different docs
    low_overlap = _row("S_lo","T_lo","DX","DY",
                       "obligation capital requirement leverage tier one", 
                       "exposure measure ifrs adjustment collateral prohibited",
    )
    high_overlap = _row("S_hi","T_hi","DX","DX", "alpha beta gamma", "alpha beta gamma")
    medium_overlap = _row("S_md","T_md","DZ","DZ", "alpha beta gamma", "alpha beta delta")

    rows = [high_overlap, medium_overlap, low_overlap]
    sel, rep = sample_xref_rows(rows, n=2, mode="hard_enriched", seed=13)

    picked_ids = { (r["SourceID"], r["TargetID"]) for r in sel }
    assert ("S_lo","T_lo") in picked_ids  # prefer low overlap, diff docs


def test_mixed_difficulty_unchanged_smoke():
    rows = [
        _row("A","B","D1","D2","a b c d e f g", "h i j k l m n"),
        _row("C","D","D1","D1","a b c", "a b c"),
    ]
    sel, rep = sample_xref_rows(rows, n=2, mode="mixed_difficulty", seed=7)
    assert rep["sampling_mode"] == "mixed_difficulty"
