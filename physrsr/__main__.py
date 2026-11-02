"""Primary CLI alias; an explicit subcommand is required for any work."""
from .active.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
