from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from evogrid.evaluation.acceptance import audit_run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved experiment outputs against EvoGrid guardrails.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    reports = [audit_run_directory(path).to_dict() for path in args.run_dirs]
    payload = {"schema_version": 1, "reports": reports, "passed": all(report["passed"] for report in reports)}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
