# ChemX Test Set

This directory is the local cache for benchmark inputs used in Step 1.

Expected generated files:

- `raw_examples.jsonl`: serialized Hugging Face rows for audit/debugging.
- `gold.csv`: normalized reference table used by `scripts/evaluate_extraction.py`.
- `manifest.json`: download summary and schema hints.
- `articles/`: article PDFs/text files copied or downloaded from the dataset rows.

The generated artifacts are ignored by git because PDFs and benchmark data can be large
or redistribution-restricted. Recreate them locally with:

```powershell
.\.venv\Scripts\python.exe scripts\download_chemx_test_set.py
```

Note: the public Hugging Face layout exposes domain datasets such as
`ai-chem/Oxazolidinones` and `ai-chem/Benzimidazoles`. The download script keeps
`ai-chem/chemx` as the requested default and falls back to `ai-chem/<domain>` when
the aggregate dataset is unavailable. Oxazolidinones currently exposes a `train`
split, so the script records the actual split in `manifest.json`.

The Oxazolidinones rows contain DOI/PDF identifiers, not article PDF bytes or
direct PDF URLs. The script writes `article_references.csv` for source audit and
copies/downloads article files only when the dataset row contains a real file,
URL, or long text payload.
