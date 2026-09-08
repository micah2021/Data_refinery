# Schema validation

This directory provides a lightweight schema contract check for `nigeria.db`.

## Run locally

```bash
python validation/validate_schema.py
```

To validate a custom database path:

```bash
DB_PATH=/path/to/nigeria.db python validation/validate_schema.py
```

## What is validated

- Required tables listed in `validation/schema.yaml`
- Required columns for each table
- Optional non-empty table checks (`non_empty_tables` in `schema.yaml`)
