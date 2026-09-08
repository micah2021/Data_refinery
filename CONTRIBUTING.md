# Contributing

Thanks for contributing to Data Refinery.

## Development setup

1. Use Python 3.10.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Create local environment variables:

```bash
cp .env.example .env
```

Update `.env` with local credentials as needed.

## Run tests and checks

```bash
flake8 tests scripts validation setup_db.py
pytest -q
python validation/validate_schema.py
python scripts/check_no_secrets.py
```

CI runs these checks for pull requests. PRs should pass linting, tests, schema validation, and secret scanning before merge.

## Proposing schema changes

- Update `nigeria_db_schema.sql`
- Update `validation/schema.yaml` so CI validates new required tables/columns
- Add or update tests in `tests/` to cover schema behavior
- Document migration/compatibility notes in the PR description
