from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from obliqaxref.utils.io import read_yaml

load_dotenv()


class PathsConfig(BaseModel):
    input_dir: str = Field(..., description="Raw input location (or workspace input).")
    work_dir: str = Field(..., description="Working directory for intermediate artifacts.")
    output_dir: str = Field(..., description="Final outputs (datasets, tables, reports).")
    curate_output_dir: str | None = Field(
        None,
        description="Curation outputs (IR runs, curated items). Defaults to output_dir if not specified.",
    )


class AdapterConfig(BaseModel):
    corpus: str = Field(..., description="fsra | ukfin")
    manifest_path: str | None = Field(
        None, description="Optional manifest listing which docs to include (UKFin)."
    )
    max_docs: int | None = Field(None, description="Optional cap for quick subsets.")
    passage_unit_policy: str = Field(
        "canonical", description="Segmentation/unit policy identifier."
    )


class GenerationConfig(BaseModel):
    methods: list[str] = Field(default_factory=lambda: ["dpel", "schema"])
    personas: list[str] = Field(default_factory=lambda: ["basic", "professional"])
    max_edges: int = 200
    qas_per_edge: int = 1
    llm_backend: str = Field("none", description="none | openai | azure | ...")
    temperature: float = 0.2
    no_citations: bool = Field(
        True,
        description=(
            "When True, the generation prompt forbids rule numbers, section numbers, article numbers, "
            "paragraph numbers, passage identifiers, and document codes in both the QUESTION and ANSWER prose. "
            "Bracketed evidence tags [#SRC:...]/[#TGT:...] are mandatory and are not affected by this flag."
        ),
    )
    no_citations_in_question: bool = Field(
        True,
        description=(
            "When True, the generation prompt instructs the model not to include citation markers, "
            "rule/section numbers, article numbers, passage identifiers, or document codes in the QUESTION. "
            "Answer evidence tags [#SRC:...]/[#TGT:...] are still required."
        ),
    )
    citation_leakage_action: str = Field(
        "keep",
        description=(
            "Action to take when citation leakage is detected in a generated question. "
            "Allowed values: 'keep' (annotate only), 'filter' (remove from output), "
            "'separate' (write to citation_explicit.qa.jsonl instead of the main file)."
        ),
    )
    dual_anchors_mode: str = Field(
        "always",
        description=(
            "Controls the dual-anchor requirement for SCHEMA generation. "
            "'always': every question must depend on at least one concrete element from each passage; "
            "'freeform_only': soft guidance applied only when no structured answer spans are present; "
            "'off': no dual-anchor constraint."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        valid_actions = {"keep", "filter", "separate"}
        if self.citation_leakage_action not in valid_actions:
            raise ValueError(
                f"citation_leakage_action must be one of {sorted(valid_actions)!r}, "
                f"got {self.citation_leakage_action!r}"
            )
        valid_modes = {"always", "freeform_only", "off"}
        if self.dual_anchors_mode not in valid_modes:
            raise ValueError(
                f"dual_anchors_mode must be one of {sorted(valid_modes)!r}, "
                f"got {self.dual_anchors_mode!r}"
            )


class IRAgreementConfig(BaseModel):
    class XRefExpansionConfig(BaseModel):
        seed_k: int = Field(20, description="Number of seed results to expand per query")
        final_k: int = Field(20, description="Number of expanded results to keep per query")
        expansion_direction: str = Field(
            "both",
            description="Cross-reference expansion direction: outgoing | incoming | both",
        )
        expansion_weight: float = Field(
            0.8,
            description="Multiplier applied to parent scores for expanded neighbours",
        )
        neighbour_score_mode: str = Field(
            "max",
            description="How to aggregate neighbour scores reached from multiple parents: max | sum",
        )
        max_expanded_per_seed: int | None = Field(
            None,
            description="Optional cap on expanded neighbours added for each seed passage",
        )

        def model_post_init(self, __context: Any) -> None:
            valid_directions = {"outgoing", "incoming", "both"}
            if self.expansion_direction not in valid_directions:
                raise ValueError(
                    "expansion_direction must be one of "
                    f"{sorted(valid_directions)!r}, got {self.expansion_direction!r}"
                )
            valid_modes = {"max", "sum"}
            if self.neighbour_score_mode not in valid_modes:
                raise ValueError(
                    "neighbour_score_mode must be one of "
                    f"{sorted(valid_modes)!r}, got {self.neighbour_score_mode!r}"
                )
            if self.seed_k <= 0:
                raise ValueError("seed_k must be positive")
            if self.final_k <= 0:
                raise ValueError("final_k must be positive")
            if self.expansion_weight < 0:
                raise ValueError("expansion_weight must be non-negative")
            if self.max_expanded_per_seed is not None and self.max_expanded_per_seed < 0:
                raise ValueError("max_expanded_per_seed must be non-negative or null")

    top_k: int = 20
    keep_threshold: int = 4  # NEW
    judge_threshold: int = 3  # NEW
    drop_threshold: int = 2  # NEW
    agreement_threshold: float = 0.8  # For backward compatibility
    retrievers: list[str] = Field(default_factory=lambda: ["bm25"])
    ir_method: str = Field(
        "majority_voting",
        description="majority_voting | weighted_voting | rrf_voting | confidence_voting",
    )
    ir_weights: dict[str, float] | None = Field(
        None, description="Per-run weights for weighted_voting"
    )
    rrf_k: int = Field(60, description="k parameter for RRF voting")
    xref_expansion: XRefExpansionConfig = Field(default_factory=XRefExpansionConfig)


class JudgeConfig(BaseModel):
    enabled: bool = False
    llm_backend: str = "none"
    score_threshold: float = 7.0
    borderline_band: tuple[float, float] = (6.5, 7.5)
    adaptive_repeats: int = 2


class CurationConfig(BaseModel):
    ir_agreement: IRAgreementConfig = Field(default_factory=IRAgreementConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    final_export_basis: str = Field(
        "dependency_valid",
        description="Final compatibility export basis: dependency_valid | answer_valid",
    )

    def model_post_init(self, __context: Any) -> None:
        valid = {"dependency_valid", "answer_valid"}
        if self.final_export_basis not in valid:
            raise ValueError(
                f"final_export_basis must be one of {sorted(valid)!r}, "
                f"got {self.final_export_basis!r}"
            )


class EvalConfig(BaseModel):
    ks: list[int] = Field(default_factory=lambda: [10, 20])
    citation_diagnostics: bool = True


class ReportConfig(BaseModel):
    intrinsic: bool = True
    human_audit_sample: int = 80
    render_html: bool = True


class PilotConfig(BaseModel):
    pilot_mode: bool = False
    pilot_n_xrefs_per_corpus: int = 50
    pilot_random_seed: int = 13
    pilot_output_suffix: str = "pilot"


class SamplingConfig(BaseModel):
    sampling_mode: str = Field(
        "random",
        description=(
            "Cross-reference row sampling strategy. "
            "'random': uniform random sample (default, backward-compatible). "
            "'low_overlap': prefer source-target pairs with low lexical overlap. "
            "'multi_ref_source': prefer source passages referencing multiple targets. "
            "'target_definition_or_condition': prefer targets with definition/condition cues. "
            "'mixed_difficulty': stratified sample across all difficulty buckets."
        ),
    )
    max_jaccard_for_low_overlap: float = Field(
        0.15,
        description=(
            "Jaccard similarity threshold below which a source-target pair is considered "
            "'low overlap'. Used by low_overlap and mixed_difficulty modes."
        ),
    )
    mixed_difficulty_bucket_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "low_overlap": 0.30,
            "multi_ref": 0.20,
            "definition_condition": 0.30,
            "random": 0.20,
        },
        description=(
            "Relative weights for each bucket in mixed_difficulty mode. "
            "Keys: low_overlap, multi_ref, definition_condition, random. "
            "Weights are normalized internally to sum to 1.0."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        _valid = {"random", "low_overlap", "multi_ref_source", "target_definition_or_condition", "mixed_difficulty", "hard_enriched"}
        if self.sampling_mode not in _valid:
            raise ValueError(
                f"sampling_mode must be one of {sorted(_valid)!r}, got {self.sampling_mode!r}"
            )


class RunConfig(BaseModel):
    run_id: str = "dev"
    paths: PathsConfig
    adapter: AdapterConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    reporting: ReportConfig = Field(default_factory=ReportConfig)
    pilot: PilotConfig = Field(default_factory=PilotConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


def load_config(path: str | Path) -> RunConfig:
    raw: dict[str, Any] = read_yaml(path)
    return RunConfig.model_validate(raw)
