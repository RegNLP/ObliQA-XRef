"""
Tests for detect_citation_leakage() — obliqaxref.generate.common.validate

Covers:
  - Positive cases: all supported leakage_type categories
  - Negative cases: false-positive protection (dates, percentages, monetary
    values, plain quantities, ordinary prose)

Run with:
    pytest tests/test_citation_leakage.py -v
"""
from __future__ import annotations

import pytest

from obliqaxref.generate.common.validate import detect_citation_leakage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(question: str, reference_text: str | None = None) -> dict:
    return detect_citation_leakage(question, reference_text=reference_text)


def _leaked(question: str, reference_text: str | None = None) -> bool:
    return _result(question, reference_text)["has_leakage"]


def _types(question: str, reference_text: str | None = None) -> list[str]:
    return _result(question, reference_text)["leakage_type"]


def _spans(question: str, reference_text: str | None = None) -> list[str]:
    return _result(question, reference_text)["matched_spans"]


# ---------------------------------------------------------------------------
# Positive cases — citation leakage SHOULD be detected
# ---------------------------------------------------------------------------

class TestRuleRef:
    def test_rule_dotted_three_levels(self):
        q = "What does Rule 3.2.1 require of authorised firms?"
        assert _leaked(q)
        assert "rule_ref" in _types(q)
        assert any("3.2.1" in s for s in _spans(q))

    def test_rule_lowercase(self):
        assert _leaked("What obligations arise under rule 3.2.1?")
        assert "rule_ref" in _types("What obligations arise under rule 3.2.1?")

    def test_rules_plural(self):
        q = "Under Rules 3.2.1 and 3.2.2, what must a firm disclose?"
        assert _leaked(q)
        assert "rule_ref" in _types(q)
        # Primary span captured
        assert any("3.2.1" in s for s in _spans(q))

    def test_rule_two_levels(self):
        # Rule 3.2 (two-level dotted) is still a citation
        assert _leaked("What does Rule 3.2 say about notifications?")
        assert "rule_ref" in _types("What does Rule 3.2 say about notifications?")

    def test_rule_without_number_not_flagged(self):
        # "rule" as a common word with no number should NOT match rule_ref
        assert "rule_ref" not in _types("The general rule is that firms must act honestly.")


class TestSectionRef:
    def test_section_with_subsection_parens(self):
        q = "What is required under Section 58(2) of the PRA Rulebook?"
        assert _leaked(q)
        assert "section_ref" in _types(q)
        assert any("58" in s for s in _spans(q))

    def test_section_plain_number(self):
        q = "Does Section 12 apply to overseas branches?"
        assert _leaked(q)
        assert "section_ref" in _types(q)

    def test_section_dotted(self):
        assert _leaked("Section 3.2.1 sets out the obligation.")
        assert "section_ref" in _types("Section 3.2.1 sets out the obligation.")

    def test_section_lowercase(self):
        assert _leaked("As required by section 4(1), firms must notify.")
        assert "section_ref" in _types("As required by section 4(1), firms must notify.")


class TestArticleRef:
    def test_article_plain(self):
        q = "How does Article 12 define 'eligible counterparty'?"
        assert _leaked(q)
        assert "article_ref" in _types(q)

    def test_article_with_parens(self):
        assert _leaked("Under Article 3(1), what are the conditions?")
        assert "article_ref" in _types("Under Article 3(1), what are the conditions?")

    def test_articles_plural(self):
        assert _leaked("Articles 5 and 6 cover reporting obligations.")
        assert "article_ref" in _types("Articles 5 and 6 cover reporting obligations.")


class TestParagraphRef:
    def test_paragraph_with_parens(self):
        q = "What conditions are set out in paragraph (3)?"
        assert _leaked(q)
        assert "paragraph_ref" in _types(q)

    def test_paragraph_plain_number(self):
        assert _leaked("The obligation in paragraph 3 applies to all firms.")
        assert "paragraph_ref" in _types("The obligation in paragraph 3 applies to all firms.")

    def test_paragraphs_plural(self):
        assert _leaked("Both paragraphs 3 and 4 must be satisfied.")
        assert "paragraph_ref" in _types("Both paragraphs 3 and 4 must be satisfied.")


class TestSubsectionRef:
    def test_subsection_plain(self):
        q = "What exemptions are listed in subsection 4?"
        assert _leaked(q)
        assert "subsection_ref" in _types(q)

    def test_subsection_with_letter(self):
        assert _leaked("Under subsection 3(a), what is required?")
        assert "subsection_ref" in _types("Under subsection 3(a), what is required?")


class TestPartRef:
    def test_part_capital(self):
        q = "Which firms must comply with the requirements in Part 2?"
        assert _leaked(q)
        assert "part_ref" in _types(q)

    def test_part_with_letter_suffix(self):
        assert _leaked("The obligation falls under Part 3A of the Rulebook.")
        assert "part_ref" in _types("The obligation falls under Part 3A of the Rulebook.")

    def test_part_lowercase_not_flagged(self):
        # "part of" in plain prose — no capital P before a number
        assert "part_ref" not in _types("This forms part of the general framework.")


class TestChapterRef:
    def test_chapter_capital(self):
        q = "What does Chapter 5 say about record-keeping obligations?"
        assert _leaked(q)
        assert "chapter_ref" in _types(q)


class TestScheduleRef:
    def test_schedule_capital(self):
        q = "What information must be included as set out in Schedule 1?"
        assert _leaked(q)
        assert "schedule_ref" in _types(q)

    def test_schedule_dotted(self):
        assert _leaked("See Schedule 2.1 for the required disclosures.")
        assert "schedule_ref" in _types("See Schedule 2.1 for the required disclosures.")


class TestCorpusCode:
    def test_gen_code(self):
        q = "Under GEN 2.1.3, what must a firm ensure?"
        assert _leaked(q)
        assert "corpus_code" in _types(q)
        assert any("GEN" in s for s in _spans(q))

    def test_cob_code(self):
        q = "What disclosure requirements apply under COB 4.5.1?"
        assert _leaked(q)
        assert "corpus_code" in _types(q)

    def test_aml_code(self):
        q = "What does AML 3.2.1 require regarding customer due diligence?"
        assert _leaked(q)
        assert "corpus_code" in _types(q)

    def test_four_char_code(self):
        assert _leaked("COBS 11.2.1 sets out best-execution requirements.")
        assert "corpus_code" in _types("COBS 11.2.1 sets out best-execution requirements.")

    def test_two_level_code(self):
        # Only two dotted levels is still a corpus reference
        assert _leaked("Under PRIN 2.1, firms must act with integrity.")
        assert "corpus_code" in _types("Under PRIN 2.1, firms must act with integrity.")


class TestReferenceTextMatch:
    def test_exact_reference_text(self):
        q = "What is required under the Senior Manager Conduct Rules?"
        ref = "Senior Manager Conduct Rules"
        result = detect_citation_leakage(q, reference_text=ref)
        assert result["has_leakage"] is True
        assert "reference_text" in result["leakage_type"]

    def test_normalized_reference_text(self):
        # Normalisation: punctuation stripped, case insensitive
        q = "What obligations arise from the senior manager conduct rules?"
        ref = "Senior Manager Conduct Rules."
        result = detect_citation_leakage(q, reference_text=ref)
        assert result["has_leakage"] is True
        assert "reference_text" in result["leakage_type"]

    def test_reference_text_not_in_question(self):
        q = "What must a firm do when onboarding a new client?"
        ref = "Senior Manager Conduct Rules"
        assert not _leaked(q, ref)

    def test_reference_text_none(self):
        # Passing None should not raise and should not affect other results
        q = "What must a firm do when onboarding a new client?"
        result = detect_citation_leakage(q, reference_text=None)
        assert "reference_text" not in result["leakage_type"]


class TestMultipleLeakageTypes:
    def test_section_and_corpus_code(self):
        q = "Under Section 3 and GEN 2.1.3, what must firms report?"
        result = detect_citation_leakage(q)
        assert result["has_leakage"] is True
        assert "section_ref" in result["leakage_type"]
        assert "corpus_code" in result["leakage_type"]

    def test_rule_and_schedule(self):
        q = "Does Rule 3.2.1 override the requirements in Schedule 1?"
        result = detect_citation_leakage(q)
        assert result["has_leakage"] is True
        assert "rule_ref" in result["leakage_type"]
        assert "schedule_ref" in result["leakage_type"]


class TestReturnStructure:
    def test_all_keys_present_when_leaked(self):
        result = detect_citation_leakage("Under Rule 3.2.1, what applies?")
        assert set(result.keys()) == {"has_leakage", "matched_patterns", "matched_spans", "leakage_type"}
        assert isinstance(result["has_leakage"], bool)
        assert isinstance(result["matched_patterns"], list)
        assert isinstance(result["matched_spans"], list)
        assert isinstance(result["leakage_type"], list)

    def test_all_keys_present_when_clean(self):
        result = detect_citation_leakage("What must a firm do?")
        assert set(result.keys()) == {"has_leakage", "matched_patterns", "matched_spans", "leakage_type"}
        assert result["has_leakage"] is False
        assert result["matched_patterns"] == []
        assert result["matched_spans"] == []
        assert result["leakage_type"] == []

    def test_empty_string(self):
        result = detect_citation_leakage("")
        assert result["has_leakage"] is False

    def test_whitespace_only(self):
        result = detect_citation_leakage("   ")
        assert result["has_leakage"] is False


# ---------------------------------------------------------------------------
# Negative cases — should NOT be flagged (false-positive protection)
# ---------------------------------------------------------------------------

class TestFalsePositiveDates:
    def test_full_date(self):
        assert not _leaked("What filings are due by 28 April 2026?")

    def test_year_only(self):
        assert not _leaked("The Financial Services Act 2012 introduced new requirements.")

    def test_year_range(self):
        assert not _leaked("Data from 2020 to 2024 must be retained.")


class TestFalsePositivePercentages:
    def test_decimal_percentage(self):
        assert not _leaked("A firm holding more than 3.5% of capital must notify.")

    def test_integer_percentage(self):
        assert not _leaked("The threshold is 25% of total voting rights.")

    def test_high_decimal_percentage(self):
        assert not _leaked("The ratio must not exceed 12.5% of total assets.")


class TestFalsePositiveMonetary:
    def test_gbp(self):
        assert not _leaked("Firms with assets exceeding £12.50 million are in scope.")

    def test_gbp_large(self):
        assert not _leaked("The firm paid a fine of £2.3 million.")

    def test_currency_code_gbp(self):
        assert not _leaked("The fee is GBP 3.5 per transaction.")

    def test_currency_code_usd(self):
        assert not _leaked("Minimum capital is USD 1.5 million.")

    def test_currency_code_eur(self):
        assert not _leaked("A fee of EUR 2.5 applies to each application.")

    def test_currency_code_aed(self):
        assert not _leaked("The penalty is AED 1.2 million.")


class TestFalsePositiveQuantities:
    def test_years(self):
        assert not _leaked("Records must be retained for 2 years after the transaction.")

    def test_business_days(self):
        assert not _leaked("The firm must respond within 3 business days.")

    def test_documents(self):
        assert not _leaked("The firm submitted 5 documents to the regulator.")

    def test_months(self):
        assert not _leaked("Notification must be given at least 6 months in advance.")

    def test_working_days(self):
        assert not _leaked("The deadline is 10 working days from the event date.")


class TestFalsePositiveProse:
    def test_plain_compliance_question(self):
        assert not _leaked(
            "What are the obligations of a firm when acting as a fiduciary for a client?"
        )

    def test_question_with_numbered_list(self):
        assert not _leaked(
            "There are 3 main obligations: 1) reporting, 2) record-keeping, 3) disclosure."
        )

    def test_rule_as_common_word(self):
        assert not _leaked("The general rule is that firms must act honestly and fairly.")

    def test_section_as_common_word(self):
        # "section" as a generic word without a following number
        assert not _leaked("In the following section, we discuss the key obligations.")

    def test_part_lowercase_in_prose(self):
        # lowercase "part" followed by a preposition, not a citation
        assert not _leaked("This forms part of the general regulatory framework.")

    def test_decimal_number_no_keyword(self):
        # A standalone dotted number with no keyword is not a citation
        assert not _leaked("The ratio of 3.5 to 1 applies in this case.")

    def test_mixed_numbers_and_dates(self):
        assert not _leaked(
            "Between 2022 and 2024, firms must maintain a 12.5% buffer and respond within 5 days."
        )
