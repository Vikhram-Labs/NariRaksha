# Dataset Pipeline Diagnostics

## Path Trace
1. **Raw Data** -> `data_ingestion.py` currently loads file text into memory but outputs unstructured mock dicts.
2. **Preprocessing** -> Non-existent. No chunking or NER is performed to extract specific BNS sections.
3. **Synthetic Generation** -> `synthetic_generation.py` completely bypasses ingestion. It uses an internal hardcoded list of risks (`RISK_CATEGORIES`) and severities (`SEVERITY_LEVELS`).
4. **Deduplication** -> Non-existent. Repeated runs will append duplicate mock strings to the `.jsonl`.
5. **Translation** -> Appends `[Lang Translation of]:` to English variants.
6. **Final JSONL Export** -> Generates 5 base mock examples, multiplies by languages to get exactly 15 records in certain tests, all with dummy tokens.

## Output Counts
- **Expected base synthetic:** 10,000+
- **Actual base synthetic:** 5
- **Expected multilingual variants:** 70,000+
- **Actual multilingual variants:** 10 (or equivalent based on manual runs)
- **Total output:** 15 records.

## Conclusion
The generation stops at step 3. The `generate_batch` method returns a statically typed python dictionary without invoking an external API or generating novel text.
