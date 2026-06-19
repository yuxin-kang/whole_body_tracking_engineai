from __future__ import annotations

import argparse
from pathlib import Path

from whole_body_tracking.rl.fast_sac.acceptance import write_acceptance_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accept or block a FastSAC run according to the paper contracts.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--expected_task_id", required=True)
    parser.add_argument("--expected_contract_id", required=True)
    parser.add_argument("--expected_recipe_contract_id", required=True)
    parser.add_argument("--expected_grsi_path", default=None)
    parser.add_argument("--expected_grsi_hash", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted, artifact_path = write_acceptance_artifacts(
        stage=args.stage,
        log_dir=Path(args.log_dir),
        expected_task_id=args.expected_task_id,
        expected_contract_id=args.expected_contract_id,
        expected_recipe_contract_id=args.expected_recipe_contract_id,
        expected_grsi_path=args.expected_grsi_path,
        expected_grsi_hash=args.expected_grsi_hash,
    )
    status = "accepted" if accepted else "blocked"
    print(f"[FastSAC Accept] {status} -> {artifact_path}")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
