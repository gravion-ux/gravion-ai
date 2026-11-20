"""Allow `python -m gravion_ai` as an alias for the `gravion` CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
