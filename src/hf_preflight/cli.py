"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from hf_preflight import __version__
from hf_preflight.core import HubError, Report, human_bytes, inspect_model

_COLOUR = {"clean": "\033[32m", "risky": "\033[33m", "blocked": "\033[31m"}
_OFF = "\033[0m"
_MARK = {"clean": " ", "risky": "!", "blocked": "x"}

# What --fail-on accepts, mapped to the finding kind it refers to.
_GATES = {
    "gated": "gated",
    "license": "license",
    "remote-code": "remote_code",
    "pickle": "pickle",
}


def _render(report: Report, *, colour: bool) -> str:
    verdict = report.severity.upper()
    if colour:
        verdict = f"{_COLOUR.get(report.severity, '')}{verdict}{_OFF}"
    lines = [verdict, f"  {report.resolved_id}"]
    if report.resolved_id != report.repo_id:
        lines[-1] += f"   (you asked for {report.repo_id})"
    lines.append("")
    for finding in report.findings:
        lines.append(f"  {_MARK.get(finding.severity, ' ')} {finding.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hf-preflight",
        description="Check a Hugging Face model before you download it.",
        epilog="exit codes: 0 clean, 1 risky, 2 blocked, 3 error",
    )
    parser.add_argument("model", help="org/name, or a huggingface.co URL")
    parser.add_argument("--revision", metavar="REF", help="branch, tag or commit")
    parser.add_argument(
        "--fail-on",
        metavar="LIST",
        help=(
            "comma-separated gates that force a non-zero exit: "
            + ", ".join(sorted(_GATES))
        ),
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-colour", action="store_true", help="disable colour")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    try:
        report = inspect_model(args.model, revision=args.revision)
    except (ValueError, HubError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps({
            "repo_id": report.repo_id,
            "resolved_id": report.resolved_id,
            "severity": report.severity,
            "gated": report.gated,
            "token_state": report.token.state,
            "token_name": report.token.name,
            "license": report.license,
            "total_bytes": report.total_bytes,
            "total_human": human_bytes(report.total_bytes),
            "weight_format": report.weight_format,
            "findings": [
                {"kind": f.kind, "severity": f.severity, "detail": f.detail}
                for f in report.findings
            ],
        }, indent=2))
    else:
        print(_render(report, colour=not args.no_colour and sys.stdout.isatty()))

    if args.fail_on:
        wanted = [g.strip() for g in args.fail_on.split(",") if g.strip()]
        unknown = [g for g in wanted if g not in _GATES]
        if unknown:
            print(
                f"error: unknown --fail-on value(s): {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 3
        for gate in wanted:
            finding = report.of_kind(_GATES[gate])
            if finding and finding.severity != "clean":
                return 2
        return 0

    return {"clean": 0, "risky": 1, "blocked": 2}[report.severity]


if __name__ == "__main__":
    raise SystemExit(main())
