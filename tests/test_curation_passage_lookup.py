import io
import json
from pathlib import Path

from obliqaxref.curate.run import load_passages


def _write_jsonl(tmp_path: Path, name: str, rows: list[dict]):
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_load_passages_indexes_multiple_id_keys(tmp_path: Path):
    # ADGM-style: pid + text
    adgm_row = {"pid": "PID-123", "text": "ADGM passage text"}
    # UKFIN-style: passage_id + passage
    ukfin_row = {"passage_id": "UK-42", "passage": "UKFIN passage text"}
    # Generic id + content
    generic_row = {"id": "GEN-9", "content": "Generic text"}

    jf = _write_jsonl(tmp_path, "passage_corpus.jsonl", [adgm_row, ukfin_row, generic_row])

    idx = load_passages(jf)

    # All keys should map
    assert "PID-123" in idx
    assert "UK-42" in idx
    assert "GEN-9" in idx

    # Stored rows should be retrievable
    assert idx["PID-123"].get("text") == "ADGM passage text"
    assert idx["UK-42"].get("passage") == "UKFIN passage text"
    assert idx["GEN-9"].get("content") == "Generic text"
