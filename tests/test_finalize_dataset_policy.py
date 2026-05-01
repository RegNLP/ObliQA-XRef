from __future__ import annotations

import json
from pathlib import Path

from obliqaxref.eval.FinalizeDataset.finalize_dataset import finalize_dataset_main
from obliqaxref.eval import cli as eval_cli


def _writelines(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_finalize_from_dependency_valid(tmp_path: Path, monkeypatch):
    # Simulate curated outputs
    curate_out = tmp_path / "runs" / "curate_adgm" / "out"
    dep_rows = [
        {"item_id": "A", "question": "qA", "gold_answer": "aA", "source_passage_id": "s1", "target_passage_id": "t1"},
        {"item_id": "B", "question": "qB", "gold_answer": "aB", "source_passage_id": "s2", "target_passage_id": "t2"},
    ]
    _writelines(curate_out / "final_dependency_valid.jsonl", dep_rows)

    # Capture outputs under a temp datasets dir
    out_dir = tmp_path / "datasets"

    # Run finalize for ADGM dependency_valid
    cwd = Path.cwd()
    try:
        # change CWD so finalize uses relative 'runs/curate_*/out'
        monkeypatch.chdir(tmp_path)
        finalize_dataset_main(out_dir=str(out_dir), corpus="adgm", cohort="dependency_valid", seed=1)
    finally:
        monkeypatch.chdir(cwd)

    # Verify combined ALL exists and has both items
    full = out_dir / "ObliQA-XRef-ADGM-ALL.jsonl"
    assert full.exists()
    lines = full.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

import json
from pathlib import Path

from obliqaxref.eval.FinalizeDataset.finalize_dataset import finalize_dataset_main


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_finalize_prefers_explicit_answer_valid_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    curate_out = tmp_path / "runs" / "curate_adgm" / "out"

    _write_jsonl(
        curate_out / "final_answer_valid.jsonl",
        [{
            "item_id": "q_answer_valid",
            "question": "Q",
            "gold_answer": "A",
            "source_passage_id": "s",
            "target_passage_id": "t",
            "method": "dpel",
        }],
    )
    _write_jsonl(
        curate_out / "curated_items.keep.jsonl",
        [{"item_id": "q_stale_keep", "decision": "KEEP"}],
    )

    finalize_dataset_main(out_dir="out", corpus="adgm", cohort="answer_valid", seed=1)

    records = _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL.jsonl")
    assert [record["item_id"] for record in records] == ["q_answer_valid"]


def test_finalize_fallback_ignores_stale_keep_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    curate_out = tmp_path / "runs" / "curate_adgm" / "out"

    _write_jsonl(
        curate_out / "curated_items.judge.jsonl",
        [
            {
                "item_id": "q_pass",
                "question": "Q",
                "gold_answer": "A",
                "source_passage_id": "s",
                "target_passage_id": "t",
                "method": "dpel",
            },
            {
                "item_id": "q_stale_keep",
                "question": "Q",
                "gold_answer": "A",
                "source_passage_id": "s",
                "target_passage_id": "t",
                "method": "schema",
            },
        ],
    )
    _write_jsonl(
        curate_out / "curate_judge" / "judge_responses_pass.jsonl",
        [{"item_id": "q_pass", "decision_qp_final": "PASS_QP"}],
    )
    _write_jsonl(
        curate_out / "curate_answer" / "answer_responses_pass.jsonl",
        [{"item_id": "q_pass", "decision_ans_final": "PASS_ANS"}],
    )
    _write_jsonl(
        curate_out / "curated_items.keep.jsonl",
        [{"item_id": "q_stale_keep", "decision": "KEEP"}],
    )

    finalize_dataset_main(out_dir="out", corpus="adgm", cohort="answer_pass", seed=1)

    records = _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL.jsonl")
    assert [record["item_id"] for record in records] == ["q_pass"]


def test_finalize_from_curate_suffix(tmp_path: Path, monkeypatch):
    # Write curated outputs under an out_<suffix> directory
    curate_out = tmp_path / "runs" / "curate_adgm" / "out_pilotX"
    dep_rows = [
        {"item_id": "SFX1", "question": "q1", "gold_answer": "a1", "source_passage_id": "s1", "target_passage_id": "t1"},
        {"item_id": "SFX2", "question": "q2", "gold_answer": "a2", "source_passage_id": "s2", "target_passage_id": "t2"},
    ]
    (curate_out).mkdir(parents=True, exist_ok=True)
    with (curate_out / "final_dependency_valid.jsonl").open("w", encoding="utf-8") as f:
        for r in dep_rows:
            import json as _json
            f.write(_json.dumps(r) + "\n")

    out_dir = tmp_path / "datasets"
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp_path)
        # Read from out_<suffix> via curate_suffix
        finalize_dataset_main(
            out_dir=str(out_dir), corpus="adgm", cohort="dependency_valid", seed=1, curate_suffix="pilotX"
        )
    finally:
        monkeypatch.chdir(cwd)

    full = out_dir / "ObliQA-XRef-ADGM-ALL.jsonl"
    assert full.exists()
    lines = full.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_stage_ir_runs_from_suffix_and_canonical_names(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = "adgm"
    gen_out = tmp_path / f"runs/generate_{corpus}/out_pilotX"
    gen_out.mkdir(parents=True, exist_ok=True)
    # Create a subset of canonical runs
    for name in [
        "bm25.trec",
        "ft_e5.trec",
        "rrf_bm25_e5.trec",
        "bm25_xref_expand.trec",
        "e5_xref_expand.trec",
        "rrf_xref_expand.trec",
        "ce_rerank_union200.trec",
    ]:
        (gen_out / name).write_text(f"{name} \n", encoding="utf-8")

    out_root = tmp_path / "ObliQA-XRef_Out_Datasets" / "pilotX"
    staged_dir = eval_cli._stage_dataset_layout(corpus, out_root, curate_suffix="pilotX")

    # Canonical files should exist
    for name in [
        "bm25.trec",
        "ft_e5.trec",
        "rrf_bm25_e5.trec",
        "bm25_xref_expand.trec",
        "e5_xref_expand.trec",
        "rrf_xref_expand.trec",
        "ce_rerank_union200.trec",
    ]:
        assert (staged_dir / name).exists(), f"missing {name}"

    # Aliases should also exist
    assert (staged_dir / "e5.trec").exists()
    assert (staged_dir / "rrf.trec").exists()


def test_finalize_staging_overwrites_stale_trecs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = "ukfin"
    gen_out = tmp_path / f"runs/generate_{corpus}/out_pilotY"
    gen_out.mkdir(parents=True, exist_ok=True)
    # Create a canonical file with known content in the source
    (gen_out / "bm25.trec").write_text("SOURCESRC\n", encoding="utf-8")

    out_root = tmp_path / "ObliQA-XRef_Out_Datasets" / "pilotY"
    staged_dir = out_root / f"ObliQA-XRef-{corpus.upper()}-ALL"
    staged_dir.mkdir(parents=True, exist_ok=True)
    # Write a stale .trec that should be removed/overwritten
    (staged_dir / "bm25.trec").write_text("STALE\n", encoding="utf-8")

    eval_cli._stage_dataset_layout(corpus, out_root, curate_suffix="pilotY")
    content = (staged_dir / "bm25.trec").read_text(encoding="utf-8")
    assert "SOURCESRC" in content and "STALE" not in content
