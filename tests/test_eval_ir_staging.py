from __future__ import annotations

from pathlib import Path

from obliqaxref.eval import cli as eval_cli


def test_ir_does_not_restaged_when_trecs_present(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = "adgm"
    out_root = tmp_path / "ObliQA-XRef_Out_Datasets" / "pilotZ"
    staged_dir = out_root / f"ObliQA-XRef-{corpus.upper()}-ALL"
    staged_dir.mkdir(parents=True, exist_ok=True)

    # Write a pre-existing staged trec
    bm25_staged = staged_dir / "bm25.trec"
    bm25_staged.write_text("STAGED\n", encoding="utf-8")

    # Prepare a different source in runs/generate_<corpus>/out to detect overwrite if it happened
    src_dir = tmp_path / f"runs/generate_{corpus}/out"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "bm25.trec").write_text("SOURCE\n", encoding="utf-8")

    # Ensure runs without forcing restage
    eval_cli._ensure_ir_runs(corpus, out_root, stage_runs=False)
    assert bm25_staged.read_text(encoding="utf-8").strip() == "STAGED"


def test_ir_stages_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = "ukfin"
    out_root = tmp_path / "ObliQA-XRef_Out_Datasets" / "rootA"

    # No staged dir yet; should stage from source
    src_dir = tmp_path / f"runs/generate_{corpus}/out"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "bm25.trec").write_text("S\n", encoding="utf-8")

    staged_dir = eval_cli._ensure_ir_runs(corpus, out_root, stage_runs=False)
    assert (staged_dir / "bm25.trec").exists()
