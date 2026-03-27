# Dependencies and reproducible installs

## Production and CI

- **Pinned set:** `requirements-lock.txt` — full transitive freeze from `pip freeze` after installing `requirements.txt` in a **clean virtualenv**.
- **Regenerate** whenever `requirements.txt` changes:
  ```bash
  python3.11 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip freeze > requirements-lock.txt
  ```
- **CI (GitHub Actions)** installs `requirements-lock.txt` first, then `requirements-dev-after-lock.txt` (pytest, ruff, pip-audit).

Use **Python 3.11** for the lockfile to match CI (`python-version: "3.11"` in `.github/workflows/ci.yml`). If your laptop only has another minor version, regenerate the lock on a 3.11 machine or in CI before merging dependency changes.

## Local development

- Quick loop: `pip install -r requirements-dev.txt` (unpinned runtime deps from `requirements.txt` plus tools).
- Parity with prod: `pip install -r requirements-lock.txt` then tooling as needed.

## Supply chain

- `pip-audit -r requirements.txt` runs in CI on direct dependencies; review and bump pins when it fails.
