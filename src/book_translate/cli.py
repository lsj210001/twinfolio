from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .pipeline import translate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="twinfolio",
        description="English EPUB in. Facing-page Chinese out.",
    )
    p.add_argument("epub", help="source .epub")
    p.add_argument("-o", "--out-dir", default=".", help="output directory (default: cwd)")
    p.add_argument("--bilingual-only", action="store_true", help="skip Chinese-only derive")
    p.add_argument("--debug", action="store_true", help="print full traceback on error")
    p.add_argument(
        "--test",
        action="store_true",
        help="translate only the first N body paragraphs (default 10); skip titles/zh",
    )
    p.add_argument(
        "--test-num",
        type=int,
        default=None,
        metavar="N",
        help="paragraph count for --test (default 10). Implies --test.",
    )
    p.add_argument(
        "--only",
        default="",
        help="comma-separated HTML names to translate (basename or zip path)",
    )
    p.add_argument(
        "--retranslate",
        default="",
        help="comma-separated HTML names to redo even if already translated",
    )
    p.add_argument(
        "--use-context",
        action="store_true",
        help="send the last N translated paragraphs as read-only context",
    )
    p.add_argument(
        "--context-paragraphs",
        type=int,
        default=0,
        metavar="N",
        help="rolling context size when --use-context (default 8)",
    )
    args = p.parse_args(argv)
    src = Path(args.epub)
    test_num = None
    if args.test or args.test_num is not None:
        test_num = args.test_num if args.test_num is not None else 10
        if test_num < 1:
            print("error: --test-num must be >= 1", file=sys.stderr)
            return 2
    try:
        outs = translate(
            src,
            Path(args.out_dir),
            bilingual_only=args.bilingual_only,
            test_num=test_num,
            only_files=args.only or None,
            retranslate=args.retranslate or None,
            use_context=args.use_context,
            context_paragraphs=args.context_paragraphs,
        )
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            cause = e.__cause__ or e.__context__
            while cause is not None:
                print(f"  caused by: {type(cause).__name__}: {cause}", file=sys.stderr)
                cause = cause.__cause__ or cause.__context__
        return 1
    for kind, path in outs.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
