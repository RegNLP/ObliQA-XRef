# Prompt Audit Report

- safe_for_overnight_run: True
- recommendation: proceed
- files_modified_by_audit: ['controlled_comparison_pilot/scripts/audit_rendered_prompts.py', 'controlled_comparison_pilot/reports/rendered_prompt_audit/*.txt', 'controlled_comparison_pilot/reports/prompt_audit_report.json', 'controlled_comparison_pilot/reports/prompt_audit_report.md']

## Prompt Files

| prompt | ok | issues |
| --- | ---: | --- |
| prompts/general_grounded_prompt.txt | True | None |
| prompts/task_aware_obliqa_prompt.txt | True | None |
| prompts/task_aware_obliqa_mp_prompt.txt | True | None |
| prompts/task_aware_xref_prompt.txt | True | None |

## Rendered Prompt Previews

| preview | ok | issues |
| --- | ---: | --- |
| reports/rendered_prompt_audit/general_obliqa_prompt_preview.txt | True | None |
| reports/rendered_prompt_audit/general_obliqa_mp_prompt_preview.txt | True | None |
| reports/rendered_prompt_audit/general_xref_adgm_prompt_preview.txt | True | None |
| reports/rendered_prompt_audit/task_aware_obliqa_prompt_preview.txt | True | None |
| reports/rendered_prompt_audit/task_aware_obliqa_mp_prompt_preview.txt | True | None |
| reports/rendered_prompt_audit/task_aware_xref_adgm_prompt_preview.txt | True | None |

## General vs Task-Aware

- general_prompt_is_dataset_neutral: True
- obliqa_task_aware_adds_obligation_guidance_only: True
- obliqa_mp_task_aware_adds_multi_passage_synthesis_guidance_only: True
- xref_task_aware_adds_citation_cross_reference_guidance_only: True
- task_aware_prompts_do_not_leak_gold_source_target_roles: True
