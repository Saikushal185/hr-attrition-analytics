"""Command-line entry point for the analytics pipeline and segment report."""
import argparse

from src import main, segments


def run():
    p = argparse.ArgumentParser(prog="hr-attrition")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analyse", help="run the full analytics pipeline")
    sub.add_parser("segments", help="print top risk segments")
    args = p.parse_args()

    if args.cmd == "analyse" and hasattr(main, "main"):
        main.main()
    elif args.cmd == "segments":
        print(segments.top_risk_segments().head(10).to_string(index=False))


if __name__ == "__main__":
    run()
