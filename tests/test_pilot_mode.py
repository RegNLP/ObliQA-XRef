# tests/test_pilot_mode.py
"""
Unit tests for pilot-run mode.

Covers:
- PilotConfig: defaults and validation
- RunConfig: pilot field present and defaults correct
- Pilot path suffix logic in generate/run.py
- Pilot path redirect logic in curate/run.py
- generate_pilot_report(): output files + report structure
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obliqaxref.config import PilotConfig, RunConfig


# =============================================================================
# PilotConfig defaults
# =============================================================================


class TestPilotConfigDefaults:
    def test_pilot_mode_off_by_default(self):
        p = PilotConfig()
        assert p.pilot_mode is False

    def test_pilot_n_xrefs_default(self):
        p = PilotConfig()
        assert p.pilot_n_xrefs_per_corpus == 50

    def test_pilot_random_seed_default(self):
        p = PilotConfig()
        assert p.pilot_random_seed == 13

    def test_pilot_output_suffix_default(self):
        p = PilotConfig()
        assert p.pilot_output_suffix == "pilot"

    def test_pilot_mode_can_be_enabled(self):
        p = PilotConfig(pilot_mode=True)
        assert p.pilot_mode is True

    def test_pilot_n_xrefs_custom(self):
        p = PilotConfig(pilot_n_xrefs_per_corpus=10)
        assert p.pilot_n_xrefs_per_corpus == 10

    def test_pilot_random_seed_custom(self):
        p = PilotConfig(pilot_random_seed=42)
        assert p.pilot_random_seed == 42

    def test_pilot_output_suffix_custom(self):
        p = PilotConfig(pilot_output_suffix="trial")
        assert p.pilot_output_suffix == "trial"


# =============================================================================
# RunConfig has pilot field
# =============================================================================


class TestRunConfigPilotField:
    def _make_run_config(self, **pilot_overrides) -> RunConfig:
        return RunConfig.model_validate(
            {
                "run_id": "test",
                "paths": {
                    "input_dir": "/tmp/in",
                    "work_dir": "/tmp/work",
                    "output_dir": "/tmp/out",
                },
                "adapter": {"corpus": "ukfin"},
                "pilot": pilot_overrides,
            }
        )

    def test_default_pilot_config_present(self):
        cfg = self._make_run_config()
        assert isinstance(cfg.pilot, PilotConfig)

    def test_pilot_mode_off_in_run_config_by_default(self):
        cfg = self._make_run_config()
        assert cfg.pilot.pilot_mode is False

    def test_pilot_mode_on_in_run_config(self):
        cfg = self._make_run_config(pilot_mode=True)
        assert cfg.pilot.pilot_mode is True

    def test_pilot_n_propagated(self):
        cfg = self._make_run_config(pilot_n_xrefs_per_corpus=25)
        assert cfg.pilot.pilot_n_xrefs_per_corpus == 25

    def test_pilot_seed_propagated(self):
        cfg = self._make_run_config(pilot_random_seed=99)
        assert cfg.pilot.pilot_random_seed == 99

    def test_pilot_suffix_propagated(self):
        cfg = self._make_run_config(pilot_output_suffix="quick")
        assert cfg.pilot.pilot_output_suffix == "quick"

    def test_model_copy_preserves_pilot(self):
        cfg = self._make_run_config(pilot_mode=True, pilot_n_xrefs_per_corpus=10)
        patched = cfg.model_copy(
            update={"pilot": cfg.pilot.model_copy(update={"pilot_output_suffix": "x"})}
        )
        assert patched.pilot.pilot_mode is True
        assert patched.pilot.pilot_n_xrefs_per_corpus == 10
        assert patched.pilot.pilot_output_suffix == "x"


# =============================================================================
# Pilot path suffix logic
# =============================================================================


class TestPilotPathSuffixLogic:
    """Validate the path-redirection logic used in generate/run.py and curate/run.py."""

    def _pilot_out_dir(self, base: str, suffix: str) -> Path:
        """Mirrors the expression used in generate/run.py."""
        base_p = Path(base)
        return base_p.parent / f"{base_p.name}_{suffix}"

    def test_suffix_appended_to_name(self):
        result = self._pilot_out_dir("/runs/generate_adgm/out", "pilot")
        assert result == Path("/runs/generate_adgm/out_pilot")

    def test_custom_suffix(self):
        result = self._pilot_out_dir("/runs/generate_adgm/out", "trial")
        assert result == Path("/runs/generate_adgm/out_trial")

    def test_does_not_modify_original_path(self):
        base = Path("/runs/generate_adgm/out")
        result = self._pilot_out_dir(str(base), "pilot")
        assert result != base

    def test_nested_path(self):
        result = self._pilot_out_dir("/a/b/c/out", "pilot")
        assert result == Path("/a/b/c/out_pilot")


# =============================================================================
# generate_pilot_report(): file outputs and report structure
# =============================================================================


class TestGeneratePilotReport:
    @pytest.fixture()
    def out_dir(self, tmp_path: Path) -> Path:
        """Create a minimal pilot output directory with fixture files."""
        d = tmp_path / "pilot_out"
        d.mkdir()

        # stats/generate_report.json
        stats_dir = d / "stats"
        stats_dir.mkdir()
        gen_report = {
            "run_id": "test",
            "pilot_mode": True,
            "pilot_n_xrefs_per_corpus": 20,
            "n_xref_rows_loaded": 20,
            "n_dpel_qas": 3,
            "n_schema_qas": 2,
            "dpel_stats": {
                "generated_total": 5,
                "citation_leakage_count": 1,
            },
            "schema_stats": {
                "generated_total": 4,
                "citation_leakage_count": 0,
            },
        }
        (stats_dir / "generate_report.json").write_text(
            json.dumps(gen_report), encoding="utf-8"
        )

        # DPEL QA file
        dpel_dir = d / "dpel"
        dpel_dir.mkdir()
        dpel_qas = [
            {
                "qa_uid": "q1",
                "method": "DPEL",
                "question": "What is the requirement?",
                "expected_answer": "The requirement is X.",
                "citation_leakage": False,
                "citation_leakage_matches": [],
                "citation_leakage_types": [],
            },
            {
                "qa_uid": "q2",
                "method": "DPEL",
                "question": "See rule 3.1 for more info.",
                "expected_answer": "Answer referencing §3.1.",
                "citation_leakage": True,
                "citation_leakage_matches": ["rule 3.1"],
                "citation_leakage_types": ["section_number"],
            },
        ]
        with open(dpel_dir / "dpel.qa.jsonl", "w", encoding="utf-8") as fh:
            for qa in dpel_qas:
                fh.write(json.dumps(qa) + "\n")

        # SCHEMA QA file (empty — valid)
        schema_dir = d / "schema"
        schema_dir.mkdir()
        (schema_dir / "schema.qa.jsonl").write_text("", encoding="utf-8")

        # curate stats.json
        curate_stats = {
            "total_items": 5,
            "ir_difficulty_label_counts": {
                "easy": 2,
                "medium": 1,
                "hard": 1,
                "source_only": 0,
                "target_only": 1,
                "neither": 0,
            },
            "final_benchmark": {"total_final": 4, "total_hard": 2},
            "answer_validation": {
                "pass_ans_count": 3,
                "drop_ans_count": 1,
                "low_consensus_count": 0,
            },
        }
        (d / "stats.json").write_text(json.dumps(curate_stats), encoding="utf-8")

        # judge pass / all items
        judge_pass = [
            {"item_id": "i1", "question": "Q1?", "ir_difficulty_label": "easy"},
            {"item_id": "i2", "question": "Q2?", "ir_difficulty_label": "hard"},
        ]
        with open(d / "judge_responses_pass.jsonl", "w", encoding="utf-8") as fh:
            for it in judge_pass:
                fh.write(json.dumps(it) + "\n")

        judge_all = judge_pass + [
            {"item_id": "i3", "question": "Q3?", "ir_difficulty_label": "neither"}
        ]
        with open(d / "curated_items.judge.jsonl", "w", encoding="utf-8") as fh:
            for it in judge_all:
                fh.write(json.dumps(it) + "\n")

        # final_hard.jsonl
        hard_items = [
            {
                "item_id": "i2",
                "question": "Q2?",
                "ir_difficulty_label": "hard",
                "difficulty_tier": "challenging",
            }
        ]
        with open(d / "final_hard.jsonl", "w", encoding="utf-8") as fh:
            for it in hard_items:
                fh.write(json.dumps(it) + "\n")

        return d

    def test_json_report_created(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        assert (out_dir / "pilot_report.json").exists()

    def test_md_report_created(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        assert (out_dir / "pilot_report.md").exists()

    def test_examples_jsonl_created(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        assert (out_dir / "pilot_examples.jsonl").exists()

    def test_report_has_n_xrefs_sampled(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        assert report["n_xrefs_sampled"] == 20

    def test_report_has_qas_generated(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        qas = report["n_qas_generated"]
        assert qas["dpel"] == 3
        assert qas["schema"] == 2
        assert qas["total"] == 5

    def test_report_has_leakage_info(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        lk = report["citation_leakage"]
        assert lk["count"] == 1  # 1 DPEL + 0 SCHEMA
        assert lk["total_checked"] == 9  # 5 + 4
        assert lk["rate"] is not None

    def test_report_has_ir_distribution(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        dist = report["curation"]["ir_difficulty_label_distribution"]
        assert dist["easy"] == 2
        assert dist["hard"] == 1

    def test_report_has_judge_pass_info(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        judge = report["curation"]["judge"]
        assert judge["pass"] == 2
        assert judge["total"] == 3

    def test_report_has_answer_validation(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        av = report["curation"]["answer_validation"]
        assert av["pass"] == 3
        assert av["total"] == 4

    def test_examples_contain_leakage_failure(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        failures = report["examples"]["citation_leakage_failures"]
        assert len(failures) >= 1
        assert failures[0]["qa_uid"] == "q2"

    def test_examples_contain_valid_item(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        valids = report["examples"]["valid_items"]
        assert len(valids) >= 1
        assert valids[0]["qa_uid"] == "q1"

    def test_examples_contain_hard_item(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        hard = report["examples"]["hard_items"]
        assert len(hard) >= 1
        assert hard[0]["item_id"] == "i2"

    def test_examples_contain_failed_item(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        failed = report["examples"]["failed_items"]
        assert len(failed) >= 1
        assert failed[0]["item_id"] == "i3"

    def test_examples_jsonl_has_category_field(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        lines = (out_dir / "pilot_examples.jsonl").read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(l) for l in lines if l.strip()]
        assert all("category" in ex for ex in parsed)

    def test_md_report_has_section_headers(self, out_dir: Path):
        from obliqaxref.pilot.report import generate_pilot_report

        generate_pilot_report(out_dir=out_dir)
        md = (out_dir / "pilot_report.md").read_text(encoding="utf-8")
        assert "# Pilot Run Report" in md
        assert "## Generation" in md
        assert "## Curation" in md

    def test_in_memory_generate_report_overrides_disk(self, out_dir: Path):
        """Pass generate_report directly; it should take precedence over disk file."""
        from obliqaxref.pilot.report import generate_pilot_report

        mem_report = {
            "pilot_n_xrefs_per_corpus": 99,
            "n_dpel_qas": 7,
            "n_schema_qas": 0,
            "dpel_stats": {"generated_total": 7, "citation_leakage_count": 2},
            "schema_stats": {},
        }
        generate_pilot_report(out_dir=out_dir, generate_report=mem_report)
        report = json.loads((out_dir / "pilot_report.json").read_text())
        assert report["n_xrefs_sampled"] == 99
        assert report["n_qas_generated"]["dpel"] == 7

    def test_empty_directory_does_not_crash(self, tmp_path: Path):
        """Report should not crash even when all input files are absent."""
        from obliqaxref.pilot.report import generate_pilot_report

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        generate_pilot_report(out_dir=empty_dir)
        assert (empty_dir / "pilot_report.json").exists()
        assert (empty_dir / "pilot_report.md").exists()
        assert (empty_dir / "pilot_examples.jsonl").exists()
