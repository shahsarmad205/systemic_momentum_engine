from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect an audit pack for a RUN_ID")
    parser.add_argument("--run-id", required=True, help="RUN_ID to package (output/runs/<RUN_ID>/)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: output/audit_packs/<RUN_ID>/)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    run_id = str(args.run_id).strip()
    src = root / "output" / "runs" / run_id
    if not src.exists():
        raise SystemExit(f"RUN_ID not found: {src}")

    out_dir = Path(args.out) if args.out else (root / "output" / "audit_packs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy run snapshot directory.
    dst = out_dir / "run_snapshot"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Copy live append-only logs for context.
    live_dir = root / "output" / "live"
    for name in ("pipeline.log", "execution_log.jsonl", "order_intents.jsonl"):
        p = live_dir / name
        if p.exists():
            shutil.copy2(p, out_dir / name)

    print(f"Audit pack written: {out_dir}")


if __name__ == "__main__":
    main()

