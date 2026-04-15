from __future__ import annotations

from collections.abc import Sequence


def collect_ee_body_violations(
    body_names: Sequence[str], error_z_values: Sequence[float], threshold: float
) -> list[dict[str, object]]:
    threshold = float(threshold)
    return [
        {
            "body_name": body_name,
            "error_z": float(error_z),
            "threshold": threshold,
            "triggered": float(error_z) > threshold,
        }
        for body_name, error_z in zip(body_names, error_z_values, strict=True)
    ]


def format_termination_debug_report(info: dict[str, object]) -> list[str]:
    triggered_terms = info.get("triggered_terms", [])
    term_names = ",".join(term["term"] for term in triggered_terms) if triggered_terms else "none"
    lines = [f"[TERM] env={info['env_id']} terms={term_names}"]

    for term in triggered_terms:
        term_name = term["term"]
        if term_name == "ee_body_pos":
            details = "; ".join(
                f"{violation['body_name']} error_z={violation['error_z']:.4f} threshold={violation['threshold']:.4f}"
                for violation in term["violations"]
            )
            lines.append(f"  - ee_body_pos: {details}")
            continue
        lines.append(f"  - {term_name}: error={term['error']:.4f} threshold={term['threshold']:.4f}")

    return lines
