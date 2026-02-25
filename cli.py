"""UltrasonicWeldMaster command-line interface."""
from __future__ import annotations
import sys


def main():
    if "--version" in sys.argv:
        print("UltrasonicWeldMaster v0.1.0")
        return 0
    print("CLI interface - use 'python cli.py --help' for usage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
