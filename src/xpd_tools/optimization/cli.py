"""Offline X-ray and UV-Vis evaluation command."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tiled.client import from_profile, from_uri

from .evaluation import (
    DEFAULT_PLQY_PARAMS,
    DEFAULT_SANDBOX_URI,
    DEFAULT_TILED_PROFILE,
    SANDBOX_CATALOG,
    PdfEvaluationMode,
    PdfFitConfig,
    PdfReferenceConfig,
    XrayUvvisEvaluation,
)

logger = logging.getLogger(__name__)


def evaluate_uids(
    uids: Sequence[str],
    evaluator: Callable[[str, Sequence[Mapping[str, Any]]], Sequence[Mapping]],
) -> list[dict[str, Any]]:
    """Evaluate UIDs sequentially and attach their source UID to each outcome."""
    outcomes: list[dict[str, Any]] = []
    for index, uid in enumerate(uids):
        evaluated = evaluator(uid, [{"_id": index}])
        if len(evaluated) != 1:
            raise ValueError(
                f"evaluator returned {len(evaluated)} outcomes for uid={uid!r}"
            )
        outcome = dict(evaluated[0])
        outcome["uid"] = uid
        outcomes.append(outcome)
    return outcomes


def _uids_from_file(path: str | Path) -> list[str]:
    """Read UIDs while ignoring blank lines and trailing comments."""
    return [
        uid
        for line in Path(path).read_text().splitlines()
        if (uid := line.split("#", maxsplit=1)[0].strip())
    ]


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate existing X-ray/UV-Vis optimization runs."
    )
    parser.add_argument("uids", nargs="*", help="Raw catalog run UIDs to evaluate.")
    parser.add_argument(
        "--uids-file",
        help="File of UIDs; blank lines and text after # are ignored.",
    )
    parser.add_argument(
        "--raw-profile",
        default=os.environ.get("TILED_PROFILE", DEFAULT_TILED_PROFILE),
        help="Tiled profile for raw data; ignored when --raw-uri is set.",
    )
    parser.add_argument(
        "--raw-uri",
        default=os.environ.get("TILED_URI"),
        help="Tiled URI for raw data, taking precedence over --raw-profile.",
    )
    parser.add_argument(
        "--sandbox-uri",
        default=os.environ.get("TILED_SANDBOX_URI", DEFAULT_SANDBOX_URI),
        help="Tiled URI containing the pdfstream sandbox catalog.",
    )
    parser.add_argument(
        "--pdf-references",
        required=True,
        metavar="PATH",
        help="Version-1 JSON PDF reference configuration.",
    )
    parser.add_argument(
        "--pdf-mode",
        choices=[mode.value for mode in PdfEvaluationMode],
        default=PdfEvaluationMode.PDF_FIT_OBJECTIVES.value,
        help="PDF correlation and fitting mode.",
    )
    parser.add_argument("--output", help="Optional CSV output path.")
    return parser


def _close_clients(*clients: Any) -> None:
    """Close each distinct Tiled context created by the command."""
    contexts: dict[int, Any] = {}
    for client in clients:
        context = getattr(client, "context", None)
        if context is not None:
            contexts[id(context)] = context
    for context in contexts.values():
        context.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate requested run UIDs and emit JSON, with optional CSV output."""
    parser = _parser()
    args = parser.parse_args(argv)
    uids = list(args.uids)
    if args.uids_file:
        uids.extend(_uids_from_file(args.uids_file))
    if not uids:
        parser.error("at least one UID or --uids-file entry is required")

    references = PdfReferenceConfig.from_json(args.pdf_references)
    raw_client: Any | None = None
    sandbox_root: Any | None = None
    try:
        raw_client = (
            from_uri(args.raw_uri) if args.raw_uri else from_profile(args.raw_profile)
        )
        created_sandbox = from_uri(args.sandbox_uri)
        sandbox_root = created_sandbox
        evaluator = XrayUvvisEvaluation(
            raw_client,
            created_sandbox[SANDBOX_CATALOG],
            DEFAULT_PLQY_PARAMS,
            references,
            pdf_fit_config=PdfFitConfig(args.pdf_mode),
        )
        outcomes = evaluate_uids(uids, evaluator)
        print(json.dumps(outcomes, indent=2, default=float))
        if args.output:
            pd.DataFrame(outcomes).to_csv(args.output, index=False)
            logger.info("Wrote evaluation outcomes to %s", args.output)
        return 0
    finally:
        _close_clients(raw_client, sandbox_root)


if __name__ == "__main__":
    raise SystemExit(main())
