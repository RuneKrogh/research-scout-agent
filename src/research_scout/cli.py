"""Command-line entry point."""

import sys


def main() -> None:
    query = " ".join(sys.argv[1:])
    if not query:
        print('Usage: research-scout "<research question>"')
        raise SystemExit(1)
    print("Agent not implemented yet.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
