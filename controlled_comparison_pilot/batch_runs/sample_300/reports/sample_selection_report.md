# Sample Selection Report

- sample_size: 300
- seed: 42

## obliqa
- source_rows: 2786
- eligible_rows: 2786
- selected_rows: 300
- stratification_fields: Group
- every_selected_query_has_10_retrieved_passages: True
- selected_ids: 010be7f5-8ec1-4d65-9626-d2912918abbf, 027ac86e-aa7e-4565-8750-2fe4ba83e370, 0322a129-a377-49b6-8cff-4009ee7a110e, 0331cf30-555d-4bc3-8da1-25f29ddf2a6b, 04203236-609a-4b0d-801b-076cb1148734, 0523dcb8-e703-48d1-84c2-98d315e4c0dc, 058612c4-4b09-4bba-9e01-c36610d827c4, 05af36dc-d73e-42ac-8c51-44dce8bdd8e0, 063e5bb5-e06d-4177-99dd-292bab5f881c, 066e11d7-2420-451e-bea9-b88b4ae6d790, 06bb0644-8f6c-43fb-bdb1-a77bde63af11, 07d4272f-7173-4cf9-b6f7-cae1b4c482df, 09a9de56-f712-4478-87c9-3110a569d028, 0ad4aef8-f4bc-4082-a28d-104c04e267f1, 0b1304c0-ead3-4808-9587-6fe3d21683cc, 0b30bdca-ee77-47dd-88ba-fc77466eb783, 0b3d0375-cac4-40e4-a734-0993b3ffb6cd, 0cdc1757-a9c9-434c-b1ca-17b15e736354, 0ea7fcf5-cce4-427b-805f-137f441b045d, 0fe80232-f866-467b-83f5-ec47f3f33a3b ...

| stratum | selected | population |
| --- | ---: | ---: |
| 1 | 83 | 770 |
| 10 | 30 | 281 |
| 2 | 76 | 708 |
| 3 | 60 | 558 |
| 4 | 51 | 469 |

## obliqa_mp
- source_rows: 447
- eligible_rows: 447
- selected_rows: 300
- stratification_fields: passage_count_bucket
- every_selected_query_has_10_retrieved_passages: True
- selected_ids: 0050f26e-1640-4e1a-92b4-9eb1222e9b81, 007dbc17-1b9a-4fb1-a274-8923eb777a0e, 0094bdbf-e00a-4e8b-b3de-7148844dfd2d, 017e3146-7e38-4a23-a456-457b1306dca2, 047ab632-44f5-44cb-954f-29092a375f47, 05184585-328f-491a-9c66-cb306e684198, 05bed25a-b5f2-4649-a3dd-95a1dd5a62cb, 05c61b5d-da36-44f2-bc61-3c9100c222e4, 05cda38b-dcf5-4e75-aeac-14c6fd9ad47d, 05cf4f68-af07-4f04-8721-91497a9370ad, 066ed020-b7e5-4229-bd5f-e5ac1fca6369, 076de581-8d8b-4076-945e-0754fe3a8711, 07972be3-5c7b-4a7f-9c42-4ff4c0077a00, 0821ea59-02ed-4937-9e30-935180113383, 0844ca37-54be-4a08-98e0-cb6725dace15, 0a1dfcd5-855d-4a9e-9e4a-59e9b2180691, 0acb1e99-e70e-4ea6-abfc-6dd783be972f, 0cf7c629-1ace-4220-ae69-1f9140e3040f, 0ed65563-60b7-4c8d-97ac-45c531189895, 1030b65c-963e-4247-a83e-706dd41b92e2 ...

| stratum | selected | population |
| --- | ---: | ---: |
| 2 | 219 | 326 |
| 3-4 | 73 | 109 |
| 5+ | 8 | 12 |

## xref_adgm
- source_rows: 502
- eligible_rows: 502
- selected_rows: 300
- stratification_fields: generation_method, sampling_regime
- every_selected_query_has_10_retrieved_passages: True
- selected_ids: 02b67492c978af4c, 0349b702678a33b4, 0591aa3d477da3ff, 05c4049981e21ee1, 069c4ebac76382fa, 09838b9f6c8bfde6, 09ed83603d5c9d13, 0aa43bdd4fa0aeef, 0b86a8dd09b7f6a1, 0c391a5aaaadf674, 0cbcd1ed70cdbfc4, 0d4d5d0951a0c7ba, 0dde26d5c3d14aaa, 0dfaae193077e415, 0e8344d7fde15d88, 0fb7f44dff94bc01, 0ff015d349726e19, 107e53c83dd8a087, 108e5fb362447202, 11657f7a0addd5b0 ...

| stratum | selected | population |
| --- | ---: | ---: |
| DPEL × hard_enriched | 44 | 74 |
| DPEL × mixed_difficulty | 59 | 99 |
| SCHEMA × hard_enriched | 84 | 141 |
| SCHEMA × mixed_difficulty | 113 | 188 |

## Warnings
- None
