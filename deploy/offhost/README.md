# Off-host artifact retention (free-first)

Goal: ensure run artifacts and append-only logs survive VM loss and support audits.

## Free-first options

### Option A: Object storage in your VM provider (preferred)

If you use OCI, use OCI Object Storage and upload:

- `output/runs/<RUN_ID>/`
- `output/live/execution_log.jsonl`
- `output/live/order_intents.jsonl`

Later, enable **retention rules** for immutability (WORM-like).

### Option B: GitHub Releases / private ops repo

For small audit packs, you can upload `output/audit_packs/<RUN_ID>/` to a private repo or a release asset.\n\n## WORM switch\n\n- OCI: enable bucket **retention rules** and lock them once you are confident.\n- AWS: S3 Object Lock (Object Lock itself has no direct charge; you pay storage/API).\n\n*** End Patch}]}$
