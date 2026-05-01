"""
Unified CLI for evaluation: finalize, humaneval, ir, answer, pipeline
(pipeline runs: finalize → stats → humaneval → IR → answer)
"""

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

from obliqaxref.eval.DownstreamEval import answer_gen_eval as answer_gen_mod
from obliqaxref.eval.DownstreamEval import ir_eval as ir_eval_mod
from obliqaxref.eval.Analysis.answer_quality_by_retrieval import (
    answer_quality_by_retrieval_outcome,
)
from obliqaxref.eval.Analysis.benchmark_statistics import generate_benchmark_statistics
from obliqaxref.eval.Analysis.human_audit import (
    aggregate_human_audit,
    export_human_audit_sample,
)
from obliqaxref.eval.FinalizeDataset.finalize_dataset import finalize_dataset_main
from obliqaxref.eval.HumanEval.compute import create_human_eval_combined
from obliqaxref.eval.ResourceStats.compute import main as resourcestats_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _stage_dataset_layout(corpus: str, out_dir: Path, curate_suffix: str | None = None) -> Path:
    """
    Normalize outputs to expected layout for downstream eval:
    ObliQA-XRef_Out_Datasets/
      ObliQA-XRef-{CORPUS}-ALL/
        train.jsonl test.jsonl dev.jsonl
        bm25.trec e5.trec rrf.trec ce_rerank_union200.trec
    """
    corpus_upper = corpus.upper()
    root = out_dir
    src_train = root / f"ObliQA-XRef-{corpus_upper}-ALL-train.jsonl"
    src_test = root / f"ObliQA-XRef-{corpus_upper}-ALL-test.jsonl"
    src_dev = root / f"ObliQA-XRef-{corpus_upper}-ALL-dev.jsonl"

    dst_dir = root / f"ObliQA-XRef-{corpus_upper}-ALL"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy/rename splits to expected filenames
    mapping = [
        (src_train, dst_dir / "train.jsonl"),
        (src_test, dst_dir / "test.jsonl"),
        (src_dev, dst_dir / "dev.jsonl"),
    ]
    for src, dst in mapping:
        if src.exists():
            shutil.copyfile(src, dst)
        else:
            logger.warning("Missing split file: %s", src)

    # Copy IR runs from generation outputs if present
    base_dir = Path(f"runs/generate_{corpus}/out")
    gen_dir = base_dir if not curate_suffix else base_dir.parent / f"{base_dir.name}_{curate_suffix}"

    logger.info("Staging IR runs from: %s", gen_dir)

    # Clean any existing staged .trec files to avoid stale/default overwrites
    for old in (p for p in dst_dir.glob("*.trec") if p.is_file()):
        try:
            old.unlink()
        except Exception:
            pass

    # Canonical filenames to stage (preserve exact method names)
    canonical = {
        "bm25.trec": gen_dir / "bm25.trec",
        "ft_e5.trec": gen_dir / "ft_e5.trec",
        "rrf_bm25_e5.trec": gen_dir / "rrf_bm25_e5.trec",
        "bm25_xref_expand.trec": gen_dir / "bm25_xref_expand.trec",
        "e5_xref_expand.trec": gen_dir / "e5_xref_expand.trec",
        "rrf_xref_expand.trec": gen_dir / "rrf_xref_expand.trec",
        "ce_rerank_union200.trec": gen_dir / "ce_rerank_union200.trec",
    }
    # Optional legacy aliases for convenience
    aliases = {
        "e5.trec": "ft_e5.trec",
        "rrf.trec": "rrf_bm25_e5.trec",
    }

    for dst_name, src_path in canonical.items():
        dst_path = dst_dir / dst_name
        if src_path.exists():
            shutil.copyfile(src_path, dst_path)
        else:
            logger.warning("Missing IR run: %s", src_path)

    # Write aliases if their canonical targets were staged
    for alias_name, target_name in aliases.items():
        src = dst_dir / target_name
        alias_dst = dst_dir / alias_name
        try:
            if src.exists():
                shutil.copyfile(src, alias_dst)
        except Exception:
            pass

    return dst_dir


def _ensure_ir_runs(corpus: str, out_dir: Path, *, stage_runs: bool = False, curate_suffix: str | None = None) -> Path:
        """Ensure staged dataset layout exists and contains IR runs.

        - If stage_runs is False and *.trec already exist under the staged dataset dir,
            do not restage to avoid overwriting pilot-staged runs.
        - If no *.trec exist or stage_runs is True, stage (copy) from generate outputs.
        """
        dst_dir = out_dir / f"ObliQA-XRef-{corpus.upper()}-ALL"
        # If trec files already present and not forcing restage, leave as-is
        trecs = list(dst_dir.glob("*.trec")) if dst_dir.exists() else []
        if trecs and not stage_runs:
                logger.info("IR runs already present; not restaging for %s", corpus.upper())
                return dst_dir
        # Otherwise, stage (will also clean stale trecs)
        return _stage_dataset_layout(corpus, out_dir, curate_suffix=curate_suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for ObliQA-XRef evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Humaneval subcommand
    humaneval_parser = subparsers.add_parser(
        "humaneval", help="Generate combined human evaluation CSV"
    )
    humaneval_parser.add_argument(
        "--corpus",
        default="both",
        choices=["ukfin", "adgm", "both"],
        help="Corpus to generate for (default: both)",
    )
    humaneval_parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Sample size per group (default: 5 per method/split/persona)",
    )
    humaneval_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    humaneval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: ObliQA-XRef_Out_Datasets)",
    )

    # Finalize subcommand
    finalize_parser = subparsers.add_parser("finalize", help="Generate final dataset and splits")
    finalize_parser.add_argument(
        "--corpus",
        default="both",
        choices=["ukfin", "adgm", "both"],
        help="Corpus to finalize (default: both)",
    )
    finalize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: ObliQA-XRef_Out_Datasets)",
    )
    finalize_parser.add_argument(
        "--cohort",
        default="dependency_valid",
        choices=["answer_valid", "dependency_valid", "answer_pass", "keep_judgepass"],
        help="Which curated cohort to finalize (default: dependency_valid)",
    )
    finalize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )
    finalize_parser.add_argument(
        "--curate-suffix",
        dest="curate_suffix",
        default=None,
        help="Optional suffix to read curated outputs from runs/curate_<corpus>/out_<SUFFIX> (default: out)",
    )

    # Stats subcommand (Resource statistics)
    stats_parser = subparsers.add_parser(
        "stats",
        help="Compute resource statistics and save under runs/... and ObliQA-XRef_Out_Datasets/DatasetStats/{corpus}",
    )
    stats_parser.add_argument(
        "--corpus",
        default="both",
        choices=["ukfin", "adgm", "both"],
        help="Corpus to compute stats for (default: both)",
    )

    # Pipeline subcommand
    subparsers.add_parser(
        "pipeline",
        help="Run full evaluation pipeline: finalize → stats → humaneval → IR → answer for both corpora",
    )

    # Prep subcommand (finalize → stats → humaneval → IR), no answer-gen
    prep_parser = subparsers.add_parser(
        "prep",
        help="Run finalize → stats → humaneval → IR (no answer generation)",
    )
    prep_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    prep_parser.add_argument(
        "--cohort",
        default="answer_valid",
        choices=["answer_valid", "dependency_valid", "answer_pass", "keep_judgepass"],
        help="Which curated cohort to finalize (default: answer_valid)",
    )
    prep_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting (default: 42)"
    )
    prep_parser.add_argument(
        "--sample-size", type=int, default=5, help="HumanEval sample size per group (default: 5)"
    )
    prep_parser.add_argument("--k", type=int, default=10, help="IR cutoff k (default: 10)")

    # Validate subcommand (sanity checks on splits/qrels)
    validate_parser = subparsers.add_parser(
        "validate", help="Validate finalized splits and qrels inputs"
    )
    validate_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )

    # IR evaluation subcommand
    ir_parser = subparsers.add_parser("ir", help="Run downstream IR evaluation on test split")
    ir_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    ir_parser.add_argument("--k", type=int, default=10, help="Cutoff k for metrics (default: 10)")
    ir_parser.add_argument("--root", default="ObliQA-XRef_Out_Datasets", help="Root output directory")
    ir_parser.add_argument(
        "--methods", nargs="*", default=None, help="List of method names (without .trec)"
    )
    ir_parser.add_argument(
        "--diag-samples", type=int, default=5, help="Number of Neither@k samples to print"
    )
    ir_parser.add_argument(
        "--normalize-docids",
        action="store_true",
        help="Normalize doc IDs (strip hyphens) for matching",
    )
    ir_parser.add_argument(
        "--stage-runs",
        action="store_true",
        help="Force restaging of IR runs from runs/generate_<corpus>/out[_<SUFFIX>] into --root (default: do not restage if *.trec already present)",
    )

    # Answer generation subcommand
    ans_parser = subparsers.add_parser(
        "answer", help="Run downstream answer generation on test split"
    )
    ans_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    # Answer evaluation subcommand
    ans_eval = subparsers.add_parser(
        "answer-eval", help="Evaluate generated answers (tags, length, ROUGE-L, passage overlap)"
    )
    ans_eval.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    ans_eval.add_argument("--root", default="ObliQA-XRef_Out_Datasets", help="Root output directory")
    ans_eval.add_argument(
        "--methods", nargs="*", default=None, help="Methods to evaluate (default: all)"
    )
    ans_eval.add_argument(
        "--no-gpt", action="store_true", help="Disable GPT scoring (default: enabled)"
    )
    ans_eval.add_argument("--no-nli", action="store_true", help="Disable NLI (default: enabled)")
    default_eval_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", ""
    )
    ans_eval.add_argument(
        "--model", default=default_eval_model, help="LLM deployment name for GPT scoring/NLI"
    )
    # External NLI toggle (default: enabled). Use --no-ext-nli to disable.
    ans_eval.add_argument(
        "--no-ext-nli",
        dest="ext_nli",
        action="store_false",
        help="Disable external CrossEncoder NLI (default: enabled)",
    )
    ans_eval.set_defaults(ext_nli=True)
    ans_eval.add_argument(
        "--nli-model", default="cross-encoder/nli-deberta-v3-base", help="External NLI model name"
    )

    quality_parser = subparsers.add_parser(
        "answer-quality-by-retrieval",
        help="Join answer evaluation outputs with retrieval outcome diagnostics",
    )
    quality_parser.add_argument(
        "--root", default="ObliQA-XRef_Out_Datasets", help="Root output directory"
    )
    quality_parser.add_argument(
        "--diagnostics",
        default=None,
        help="Path to retrieval_diagnostics_per_query.csv (default: <root>/retrieval_diagnostics_per_query.csv)",
    )
    quality_parser.add_argument("--out", default=None, help="Output directory (default: --root)")
    quality_parser.add_argument("--corpus", default=None, choices=["ukfin", "adgm"])
    quality_parser.add_argument(
        "--min-group-size", type=int, default=5, help="Small-group warning threshold"
    )

    audit_export = subparsers.add_parser(
        "human-audit-export", help="Export a stratified CSV sample for manual audit"
    )
    audit_export.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Input final_answer_valid JSONL/CSV files (default: runs/curate_*/out/final_answer_valid.jsonl)",
    )
    audit_export.add_argument("--out", default="ObliQA-XRef_Out_Datasets", help="Output directory")
    audit_export.add_argument("--n", type=int, default=200, help="Requested sample size")
    audit_export.add_argument("--seed", type=int, default=13, help="Random seed")
    audit_export.add_argument(
        "--stratify-persona",
        action="store_true",
        help="Include persona in sampling strata",
    )

    audit_aggregate = subparsers.add_parser(
        "human-audit-aggregate", help="Aggregate completed human audit annotation CSV files"
    )
    audit_aggregate.add_argument("--inputs", nargs="+", required=True, help="Annotation CSV files")
    audit_aggregate.add_argument(
        "--out", default="ObliQA-XRef_Out_Datasets", help="Output directory"
    )
    audit_aggregate.add_argument(
        "--min-group-size", type=int, default=5, help="Small-group warning threshold"
    )

    benchmark_stats = subparsers.add_parser(
        "benchmark-statistics", help="Generate ObliQA-XRef benchmark statistics tables"
    )
    benchmark_stats.add_argument(
        "--input",
        nargs="*",
        default=None,
        help=(
            "Final cohort JSONL/CSV files. Default: runs/curate_*/out/"
            "final_dependency_valid.jsonl, final_answer_valid.jsonl, final_answer_failed.jsonl"
        ),
    )
    benchmark_stats.add_argument(
        "--out", default="ObliQA-XRef_Out_Datasets", help="Output directory"
    )
    benchmark_stats.add_argument(
        "--min-group-size", type=int, default=5, help="Small-group warning threshold"
    )

    ans_parser.add_argument("--k", type=int, default=10, help="Top-k passages to use per query")
    ans_parser.add_argument("--root", default="ObliQA-XRef_Out_Datasets", help="Root output directory")
    ans_parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="IR methods to generate answers for (default: all)",
    )
    default_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", ""
    )
    ans_parser.add_argument(
        "--model", default=default_model, help="LLM model/deployment to use for answer generation"
    )
    # Default: use retrieved passages; allow disabling with --no-use-retrieved
    ans_parser.add_argument(
        "--no-use-retrieved",
        dest="use_retrieved",
        action="store_false",
        help="Disable using retrieved passages (default: enabled)",
    )
    ans_parser.set_defaults(use_retrieved=True)

    args = parser.parse_args()

    if args.command == "humaneval":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            create_human_eval_combined(
                c,
                sample_size=args.sample_size,
                seed=args.seed,
                output_dir=args.output_dir,
            )

    elif args.command == "finalize":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        out_dir = Path(args.output_dir) if args.output_dir else Path("ObliQA-XRef_Out_Datasets")
        for c in corpora:
            finalize_dataset_main(
                out_dir=str(out_dir),
                corpus=c,
                cohort=args.cohort,
                seed=args.seed,
                curate_suffix=args.curate_suffix,
            )
            staged_dir = _stage_dataset_layout(c, out_dir, curate_suffix=args.curate_suffix)
            logger.info("Staged dataset for downstream eval: %s", staged_dir)

    elif args.command == "stats":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            resourcestats_main(c)

    elif args.command == "pipeline":
        # Run full pipeline for both corpora
        for corpus in ["ukfin", "adgm"]:
            logger.info("\n=== PIPELINE: %s ===", corpus.upper())

            # 1) Finalize dataset
            logger.info("[1/5] Finalizing datasets...")
            out_root = Path("ObliQA-XRef_Out_Datasets")
            finalize_dataset_main(
                out_dir=str(out_root),
                corpus=corpus,
                cohort="answer_valid",
            )
            staged_dir = _stage_dataset_layout(corpus, out_root)

            # 2) Resource statistics
            logger.info("[2/5] Computing resource statistics...")
            resourcestats_main(corpus)

            # 3) Human evaluation CSV
            logger.info("[3/5] Generating human evaluation CSV...")
            create_human_eval_combined(corpus, sample_size=5)

            # 4) IR Evaluation
            logger.info("[4/5] Running IR evaluation...")
            ir_eval_cmd = [
                "python",
                "src/obliqaxref/eval/DownstreamEval/ir_eval.py",
                "--corpus",
                corpus,
                "--k",
                "10",
                "--root",
                "ObliQA-XRef_Out_Datasets",
            ]
            subprocess.run(ir_eval_cmd, check=True)

            # 5) Answer Generation
            logger.info("[5/5] Running answer generation...")
            answer_gen_cmd = [
                "python",
                "src/obliqaxref/eval/DownstreamEval/answer_gen_eval.py",
                "--corpus",
                corpus,
                "--k",
                "10",
                "--root",
                "ObliQA-XRef_Out_Datasets",
                "--method",
                "bm25",
            ]
            subprocess.run(answer_gen_cmd, check=True)

        logger.info("\n✓ Full evaluation pipeline completed for all corpora.")

    elif args.command == "prep":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        out_root = Path("ObliQA-XRef_Out_Datasets")
        for c in corpora:
            logger.info("\n=== PREP: %s ===", c.upper())
            # finalize
            finalize_dataset_main(
                out_dir=str(out_root), corpus=c, cohort=args.cohort, seed=args.seed
            )
            staged_dir = _stage_dataset_layout(c, out_root)
            # stats
            resourcestats_main(c)
            # humaneval
            create_human_eval_combined(c, sample_size=args.sample_size)
            # ir eval
            ir_eval_mod.main(
                corpus=c,
                k=args.k,
                methods=None,
                root_dir=str(out_root),
                diag_samples=5,
                normalize_docids=False,
            )
        logger.info("\n✓ Prep completed (finalize → stats → humaneval → IR)")

    elif args.command == "validate":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            c_up = c.upper()
            test_path = Path(f"ObliQA-XRef_Out_Datasets/ObliQA-XRef-{c_up}-ALL-test.jsonl")
            total = ok = 0
            missing = []
            try:
                with test_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        import json

                        o = json.loads(line)
                        total += 1
                        if o.get("source_passage_id") and o.get("target_passage_id"):
                            ok += 1
                        else:
                            missing.append(o.get("item_id"))
                logger.info(
                    "%s test: total=%d, with_both_ids=%d, missing=%d", c_up, total, ok, len(missing)
                )
            except FileNotFoundError:
                logger.warning("Missing test split for %s: %s", c_up, test_path)

    elif args.command == "ir":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            root_dir = Path(args.root)
            try:
                _ensure_ir_runs(c, root_dir, stage_runs=args.stage_runs)
            except Exception as e:
                logger.warning("IR staging check failed for %s: %s", c, e)

            ir_eval_mod.main(
                corpus=c,
                k=args.k,
                methods=args.methods,
                root_dir=str(root_dir),
                diag_samples=args.diag_samples,
                normalize_docids=args.normalize_docids,
            )

    elif args.command == "answer":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        # Ensure expected layout and TREC runs are staged under root
        root_dir = Path(args.root)
        for c in corpora:
            try:
                _stage_dataset_layout(c, root_dir)
            except Exception as e:
                logger.warning("Could not stage dataset layout for %s: %s", c, e)
        methods = args.methods or ["bm25", "e5", "rrf", "ce_rerank_union200"]
        for c in corpora:
            for m in methods:
                answer_gen_mod.main(
                    corpus=c,
                    k=args.k,
                    method=m,
                    model=args.model,
                    root_dir=str(root_dir),
                    use_retrieved=args.use_retrieved,
                )

    elif args.command == "answer-eval":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        root_dir = Path(args.root)
        from obliqaxref.eval.DownstreamEval.answer_eval import main as eval_answers_main

        methods = args.methods or ["bm25", "e5", "rrf", "ce_rerank_union200"]
        # stage layout (ensures trec/splits copied for consistency; not strictly needed here but harmless)
        for c in corpora:
            try:
                _stage_dataset_layout(c, root_dir)
            except Exception:
                pass
            eval_answers_main(
                corpus=c,
                methods=methods,
                root_dir=str(root_dir),
                use_gpt=not args.no_gpt,
                use_nli=not args.no_nli,
                model=args.model,
                use_external_nli=args.ext_nli,
                nli_model_name=args.nli_model,
            )

    elif args.command == "answer-quality-by-retrieval":
        paths = answer_quality_by_retrieval_outcome(
            root_dir=args.root,
            diagnostics_path=args.diagnostics,
            out_dir=args.out,
            corpus=args.corpus,
            min_group_size=args.min_group_size,
        )
        for label, path in paths.items():
            logger.info("Wrote %s: %s", label, path)

    elif args.command == "human-audit-export":
        paths = export_human_audit_sample(
            inputs=args.input,
            out_dir=args.out,
            n=args.n,
            seed=args.seed,
            stratify_persona=args.stratify_persona,
        )
        for label, path in paths.items():
            logger.info("Wrote %s: %s", label, path)

    elif args.command == "human-audit-aggregate":
        paths = aggregate_human_audit(
            inputs=args.inputs,
            out_dir=args.out,
            min_group_size=args.min_group_size,
        )
        for label, path in paths.items():
            logger.info("Wrote %s: %s", label, path)

    elif args.command == "benchmark-statistics":
        paths = generate_benchmark_statistics(
            inputs=args.input,
            out_dir=args.out,
            min_group_size=args.min_group_size,
        )
        for label, path in paths.items():
            logger.info("Wrote %s: %s", label, path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
