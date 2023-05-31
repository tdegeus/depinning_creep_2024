import argparse
import sys


def _parse(parser: argparse.ArgumentParser, cli_args: list[str]) -> argparse.ArgumentParser:
    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])
