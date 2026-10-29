"""Convenience launcher; the implementation lives under scripts/."""

if __package__:
    from .scripts.run_case import main
else:
    from scripts.run_case import main


if __name__ == "__main__":
    raise SystemExit(main())
