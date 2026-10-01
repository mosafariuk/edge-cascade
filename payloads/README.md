# Payloads

| Directory | What it is |
|---|---|
| `property_audit/`, `telemetry/` | Four pseudonymized illustrative samples (2 + 2) used by the tests and by paper §V-D. Any other file placed here is gitignored. |
| `synthetic-confirm/` | **The confirmation corpus behind paper Table VIII.** 1,000 + 1,000 records, seed 20261004, generated after every design choice and the in-context examples were frozen. |
| `synthetic/` | **The exploratory corpus (paper §VI-G).** 250 + 250 records, seed 20261002, used to compare guard statistics and explore generation settings. |

Both synthetic corpora are built by `bench/gen-synthetic-corpus.mjs`: the ground truth of every
record is sampled in code from a seeded generator (`metadata.synthetic_truth`, seed in
`MANIFEST.json`); a language model only writes the text; every text is then verified
mechanically (`faithful()`) to state its truth, and regenerated if it does not. They contain no
personal data. `metadata.synthetic_truth` is never shown to a model: extraction prompts read
only `raw_text` / `raw_telemetry`.

## Payload contract

Each file is one ingestion payload as described in `INGESTION.md` §1:

```
property_audit/*.json   { req_id, source, raw_text, metadata? }
telemetry/*.json        { req_id, stream_type, raw_telemetry, timestamp?, metadata? }
```

Validate any payload against the inbound contract with:

```bash
node payloads/validate-samples.mjs     # runs admit() over every sample, reports route + accept/reject
```

## What extraction leaves to code

The model returns raw primitives only; normalization and every derived field are computed in
`src/schemas.mjs` (`derive`):

- **Currency** — `£14,500`, `14500`, `£14.5k`; ambiguous ranges are escalated, not guessed.
- **Dates** — `effective_date_raw` is extracted verbatim; the year is resolved from batch
  metadata and `DD/MM` vs `MM/DD` ambiguity is flagged.
- **Percentages and discrepancy flags** — computed from the extracted values.
- **Telemetry markers** — counted in code; `requires_deep_diagnostic` is the derived conjunction.
